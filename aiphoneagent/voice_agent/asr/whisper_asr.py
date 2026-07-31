"""faster-whisper ASR client.

One shared `WhisperModel` (loaded on GPU 4, int8_float16) serves all calls.
Transcription is synchronous (CTranslate2), so we run it in a worker thread
to keep the asyncio event loop responsive.

Tuned for short phone utterances: greedy-ish beam, no cross-utterance
conditioning, and silence-hallucination suppression (no_speech / logprob
thresholds + a VAD backstop).
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Whisper hallucinates these on silence/echo (trained on subtitled video). If a
# transcript is essentially one of these, treat it as no speech and drop it.
_HALLUCINATION_SUBSTR = (
    "authorwave",
    "υπότιτλοι",            # "Subtitles ..." (Greek)
    "υποτιτλισμός",
    "προγραμματισμός υπο",
    "amara",
    "subtitle",
    "subtitles by",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "♪",
)


def _is_hallucination(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False
    return any(h in t for h in _HALLUCINATION_SUBSTR)


def _freest_gpu() -> int:
    """Pick the CUDA device with the most free VRAM (via nvidia-smi).

    Lets ASR_DEVICE_INDEX=auto run on whatever GPU has room — handy since
    GPU 4 is saturated by XTTS+e5. Falls back to 0 if nvidia-smi is absent.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        free = [int(x) for x in out.split()]
        idx = max(range(len(free)), key=free.__getitem__)
        logger.info("ASR auto-selected GPU %d (%d MiB free)", idx, free[idx])
        return idx
    except Exception:  # noqa: BLE001 - nvidia-smi missing or unparsable
        logger.warning("Could not query GPUs for auto-select; defaulting to GPU 0")
        return 0


def _cuda_device_count() -> int:
    """How many CUDA devices CTranslate2 can actually see (after CUDA_VISIBLE_DEVICES)."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001 - no CT2 CUDA build / query failed
        return 0


def _resolve_device(device: str, index: int | str, compute_type: str) -> tuple[str, list[int], str]:
    """Normalize (device, index, compute_type) into faster-whisper args.

    Validates the GPU ordinal against the count CTranslate2 actually sees so a
    bad index fails with an actionable message instead of CUDA's opaque
    "invalid device ordinal" (which crash-loops under systemd).
    """
    if device == "cpu":
        # float16 is GPU-only; CPU needs an int8/float32 type.
        if "float16" in compute_type:
            logger.info("CPU device: overriding compute_type '%s' -> 'int8'", compute_type)
            compute_type = "int8"
        return "cpu", [0], compute_type

    count = _cuda_device_count()
    if count == 0:
        raise RuntimeError(
            "ASR_DEVICE=cuda but CTranslate2 sees no CUDA devices. Check the GPU "
            "driver, that the CUDA-enabled CTranslate2 build is installed, and "
            "that CUDA_VISIBLE_DEVICES isn't empty. Set ASR_DEVICE=cpu to run on CPU."
        )

    if str(index) == "auto":
        resolved = min(_freest_gpu(), count - 1)  # clamp: nvidia-smi may show more than CT2 sees
    else:
        resolved = int(index)

    if not 0 <= resolved < count:
        raise RuntimeError(
            f"ASR_DEVICE_INDEX={index} is out of range — CTranslate2 sees {count} "
            f"CUDA device(s) (valid ordinals 0..{count - 1}). This usually means "
            f"CUDA_VISIBLE_DEVICES remaps the GPUs: if it's set to e.g. '5', that "
            f"GPU becomes ordinal 0 inside the process. Fix by either setting "
            f"ASR_DEVICE_INDEX to a valid ordinal (0..{count - 1}), using "
            f"ASR_DEVICE_INDEX=auto, or clearing CUDA_VISIBLE_DEVICES so all GPUs "
            f"are visible and the physical index (e.g. 5) is valid again."
        )
    return "cuda", [resolved], compute_type


@dataclass
class TranscriptResult:
    text: str
    language: str
    no_speech_prob: float
    avg_logprob: float
    language_probability: float = 1.0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class WhisperASR:
    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "cuda",
        device_index: int | str = 5,
        compute_type: str = "int8_float16",
        supported_languages: list[str] | None = None,
        primary_language: str = "el",
    ) -> None:
        from faster_whisper import WhisperModel

        dev, idx, ctype = _resolve_device(device, device_index, compute_type)
        logger.info("Loading faster-whisper '%s' on %s:%s (%s)", model, dev, idx, ctype)
        self._model = WhisperModel(model, device=dev, device_index=idx, compute_type=ctype)
        self._supported = set(supported_languages or [primary_language])
        self._primary = primary_language

    async def transcribe(self, pcm16_16k: np.ndarray, language: str | None = None) -> TranscriptResult:
        return await asyncio.to_thread(self._transcribe_sync, pcm16_16k, language)

    def _transcribe_sync(self, pcm16_16k: np.ndarray, language: str | None) -> TranscriptResult:
        audio = np.asarray(pcm16_16k, dtype=np.float32) / 32768.0
        segments, info = self._model.transcribe(
            audio,
            language=language,            # None => auto-detect (first turn)
            beam_size=1,                  # latency over marginal accuracy on a phone line
            condition_on_previous_text=False,
            vad_filter=True,              # backstop; primary endpointing is upstream
            vad_parameters={"min_silence_duration_ms": 300},
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            temperature=0.0,
        )
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments).strip()
        if _is_hallucination(text):
            logger.info("Dropping hallucinated transcript: %r", text)
            text = ""
        no_speech = min((s.no_speech_prob for s in segments), default=1.0)
        avg_logprob = (
            sum(s.avg_logprob for s in segments) / len(segments) if segments else -10.0
        )

        detected = info.language or self._primary
        lang_prob = float(getattr(info, "language_probability", 1.0) or 1.0)
        if detected not in self._supported:
            logger.info("Detected unsupported language '%s'; using primary '%s'", detected, self._primary)
            detected = self._primary

        return TranscriptResult(
            text=text,
            language=detected,
            no_speech_prob=no_speech,
            avg_logprob=avg_logprob,
            language_probability=lang_prob,
        )
