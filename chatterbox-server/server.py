#!/usr/bin/env python3
"""Chatterbox Multilingual TTS pilot server (V100-safe), drop-in for VoxCPM.

Thin FastAPI wrapper around resemble-ai Chatterbox Multilingual. Implements the
same endpoint contract as the VoxCPM server (speech / stream / voices / clone)
so it can take over the canonical TTS port with no caller changes. English-first;
Greek (language_id='el') is wired and ready.

V100 (Volta, sm_70) rules:
  * Run fp32 (default) or fp16 -- NEVER bfloat16 (no native bf16 on Volta).
  * TORCHDYNAMO_DISABLE=1 (same torch.compile guard the VoxCPM server needs).
"""
import os
import io
import inspect
import logging
from contextlib import asynccontextmanager

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # torch.compile guard

import torch  # noqa: F401  (import side effects / availability check)
# NOTE: torchaudio 2.11's ta.save() delegates WAV encoding to TorchCodec, which
# is not installed (and would drag in FFmpeg system libs). soundfile is already a
# dependency here — streaming.py uses it — so encode with that instead.
import soundfile as sf
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

import voices
from audio_post import postprocess, TTS_OUTPUT_SAMPLE_RATE, TTS_SPEED
from streaming import stream_pcm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("chatterbox-pilot")

# --- Config (env-overridable) ---
PORT = int(os.environ.get("CHATTERBOX_PORT", 8094))
DEVICE = os.environ.get("CHATTERBOX_DEVICE", "cuda")
T3_MODEL = os.environ.get("CHATTERBOX_T3_MODEL", "")  # empty = load default (already multilingual)
DEFAULT_LANG = os.environ.get("CHATTERBOX_LANG", "en")
CFG_WEIGHT = float(os.environ.get("CHATTERBOX_CFG_WEIGHT", 0.5))      # higher = adheres to ref + slower
EXAGGERATION = float(os.environ.get("CHATTERBOX_EXAGGERATION", 0.5))  # higher = livelier+faster+variable
# Lower temperature => deterministic => SAME speaker every call (kills per-sentence drift).
TEMPERATURE = float(os.environ.get("CHATTERBOX_TEMPERATURE", 0.6))
USE_FP16 = os.environ.get("CHATTERBOX_FP16", "false").lower() in ("1", "true", "yes")

# 23 languages supported by Chatterbox Multilingual.
LANGUAGES = {c: c for c in ["ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
                            "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
                            "sw", "tr", "zh"]}
CONTENT_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav", "opus": "audio/opus", "flac": "audio/flac"}


class State:
    model = None
    sr = None


state = State()


class TTSRequest(BaseModel):
    """OpenAI-ish TTS request (subset), shared by speech + stream."""
    input: str = Field(..., description="Text to synthesize")
    language: str = Field(default=DEFAULT_LANG, description="language_id, e.g. en / el")
    voice: str = Field(default="default", description="'default', preset/clone id, or 'none'")
    cfg_weight: float | None = Field(default=None, description="higher=adheres to ref+slower")
    exaggeration: float | None = Field(default=None, description="override expressiveness")
    temperature: float | None = Field(default=None, description="lower=same voice every call")
    speed: float | None = Field(default=None, description="post-hoc tempo; <1.0=slower")
    response_format: str = Field(default="wav")
    sample_rate: int | None = Field(default=None, description="output rate (e.g. 8000 phone); 0/None=env")


def _maybe_fp16(model):
    """Best-effort fp16 cast of the heavy backbone only. NEVER bf16 on V100."""
    if not USE_FP16 or DEVICE != "cuda":
        return
    backbone = getattr(model, "t3", None)
    if hasattr(backbone, "half"):
        try:
            backbone.half()
            logger.info("t3 backbone cast to fp16")
        except Exception as e:
            logger.warning("fp16 cast failed, staying fp32: %s", e)


def _filter_supported(method, kwargs: dict) -> dict:
    """Drop kwargs the installed Chatterbox build doesn't accept."""
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _build_gen_kwargs(req: TTSRequest) -> dict:
    """Assemble generate() kwargs (voice routing, cloning ref, pacing levers)."""
    voice = voices.select_voice(req.voice, req.input)
    ref = None if req.voice == "none" else voices.resolve_reference(voice)
    return {
        "text": req.input,
        "language_id": req.language or DEFAULT_LANG,
        "audio_prompt_path": ref,
        "cfg_weight": req.cfg_weight if req.cfg_weight is not None else CFG_WEIGHT,
        "exaggeration": req.exaggeration if req.exaggeration is not None else EXAGGERATION,
        "temperature": req.temperature if req.temperature is not None else TEMPERATURE,
    }


def synthesize(req: TTSRequest) -> bytes:
    """Run Chatterbox -> WAV/format bytes, with VoxCPM-matched post-processing."""
    raw = _build_gen_kwargs(req)
    if raw["audio_prompt_path"] is None:
        logger.warning("NO reference voice -> Chatterbox WILL invent a random speaker. "
                       "Check DEFAULT_SPEAKER_WAV exists and is readable by the service user.")
    gen = _filter_supported(state.model.generate, raw)
    if raw.get("audio_prompt_path") and "audio_prompt_path" not in gen:
        logger.warning("generate() has no audio_prompt_path param -> voice cloning is OFF "
                       "(random speaker). Chatterbox build/API mismatch.")
    logger.info("synth lang=%s ref=%s cfg=%s exag=%s '%s...'", raw["language_id"],
                raw["audio_prompt_path"], raw["cfg_weight"], raw["exaggeration"], req.input[:40])
    wav = state.model.generate(**gen)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    buf = io.BytesIO()
    # soundfile wants (frames, channels); the tensor is (channels, frames).
    sf.write(buf, wav.cpu().float().numpy().T, state.sr,
             format="WAV", subtype="PCM_16")
    spd = req.speed if req.speed is not None else TTS_SPEED
    return postprocess(buf.getvalue(), req.response_format, req.sample_rate or 0, spd)


def _load_model(cls):
    """Load the model. Pass t3_model only if requested AND this build supports it;
    otherwise load the default checkpoint (already the 23-language multilingual one)."""
    if T3_MODEL:
        try:
            return cls.from_pretrained(device=DEVICE, t3_model=T3_MODEL)
        except TypeError:
            logger.warning("this build's from_pretrained has no t3_model arg; loading default checkpoint")
    return cls.from_pretrained(device=DEVICE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    logger.info("Loading Chatterbox Multilingual on %s (t3_model=%s) ...", DEVICE, T3_MODEL or "default")
    state.model = _load_model(ChatterboxMultilingualTTS)
    state.sr = state.model.sr
    _maybe_fp16(state.model)
    voices.load_voices()
    logger.info("Chatterbox ready (sr=%s, fp16=%s)", state.sr, USE_FP16)
    yield


app = FastAPI(title="Chatterbox Multilingual TTS (pilot)", lifespan=lifespan)


# --- Health / metadata ---

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "chatterbox-multilingual", "t3_model": T3_MODEL,
            "sr": state.sr, "fp16": USE_FP16, "cfg_weight": CFG_WEIGHT,
            "exaggeration": EXAGGERATION, "voices_loaded": len(voices.list_voices())}


@app.get("/v1/audio/languages")
async def list_languages():
    return {"languages": LANGUAGES}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "tts-1", "object": "model", "owned_by": "tec-dynamics"}]}


# --- Speech (full file) ---

def _speech_response(request: TTSRequest) -> Response:
    audio = synthesize(request)
    ctype = CONTENT_TYPES.get(request.response_format, "audio/wav")
    return Response(content=audio, media_type=ctype,
                    headers={"Content-Disposition": f'attachment; filename="speech.{request.response_format}"'})


@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    try:
        return _speech_response(request)
    except Exception as e:
        logger.error("synthesis failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech/public")
async def create_speech_public(request: TTSRequest):
    return await create_speech(request)


# --- Streaming (raw PCM16, matched to VoxCPM /stream) ---

def _stream_response(request: TTSRequest) -> StreamingResponse:
    native = state.sr
    out_rate = int(request.sample_rate or TTS_OUTPUT_SAMPLE_RATE or native)
    raw = _build_gen_kwargs(request)

    def fallback() -> bytes:
        wav_req = request.model_copy(update={"response_format": "wav"})
        return synthesize(wav_req)

    gen = stream_pcm(state.model, raw, native, out_rate, fallback)
    return StreamingResponse(gen, media_type=f"audio/L16; rate={out_rate}; channels=1",
                             headers={"X-Sample-Rate": str(out_rate), "X-Audio-Format": "pcm_s16le",
                                      "Cache-Control": "no-cache"})


@app.post("/v1/audio/speech/stream")
async def create_speech_stream(request: TTSRequest):
    return _stream_response(request)


@app.post("/v1/audio/speech/stream/public")
async def create_speech_stream_public(request: TTSRequest):
    return _stream_response(request)


# --- Voices ---

@app.get("/v1/audio/voices")
async def list_voices_endpoint():
    return {"voices": voices.list_voices()}


@app.post("/v1/audio/clone")
async def clone_voice(name: str = Form(...), description: str = Form(default=""),
                      language: str = Form(default="en"), reference_text: str = Form(default=""),
                      audio: UploadFile = File(...)):
    """Clone a voice from an uploaded sample (Chatterbox needs no transcript)."""
    try:
        content = await audio.read()
        voice_id = voices.save_clone(name, content, audio.filename or "sample.wav")
        return {"success": True, "voice_id": voice_id, "name": name}
    except Exception as e:
        logger.error("clone failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {e}")


@app.delete("/v1/audio/voices/{voice_id}")
async def delete_voice_endpoint(voice_id: str):
    if not voices.delete_voice(voice_id):
        raise HTTPException(status_code=400, detail="Not found or not a deletable cloned voice")
    return {"success": True, "message": f"Voice {voice_id} deleted"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1)
