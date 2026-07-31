#!/usr/bin/env python3
"""Voice registry for the Chatterbox pilot: presets, cloned voices, resolution.

Chatterbox clones zero-shot from a reference wav (no transcript needed, unlike
VoxCPM1.5), so a voice is just a name -> wav path mapping. Mirrors the VoxCPM
server's voice contract (default / presets / cloned + lock + Greek routing).
"""
import os
import json
import time
import hashlib
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("chatterbox-pilot")

VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/tec/ai/voices"))
PRESET_VOICES_DIR = Path(os.environ.get("PRESET_VOICES_DIR", "/tec/ai/voices/presets"))
DEFAULT_SPEAKER_WAV = os.environ.get("DEFAULT_SPEAKER_WAV", "/tec/ai/voices/default_speaker.wav")
TTS_LOCK_VOICE = os.environ.get("TTS_LOCK_VOICE", "").strip()
TTS_VOICE_EL = os.environ.get("TTS_VOICE_EL", "").strip()

_voices = {}


def has_greek(text: str) -> bool:
    """True if text contains Greek script (for Greek-voice routing)."""
    return any(("Ͱ" <= ch <= "Ͽ") or ("ἀ" <= ch <= "῿") for ch in text)


def load_voices():
    """Load cloned (voices.json) + preset (<name>.wav) + default voices."""
    _voices.clear()
    vf = VOICES_DIR / "voices.json"
    if vf.exists():
        try:
            _voices.update(json.loads(vf.read_text()))
        except Exception as e:
            logger.error("voices.json load failed: %s", e)
    if PRESET_VOICES_DIR.exists():
        for wav in sorted(PRESET_VOICES_DIR.glob("*.wav")):
            _voices[wav.stem] = {"id": wav.stem, "name": wav.stem,
                                 "is_cloned": True, "sample_path": str(wav)}
    default_ref = DEFAULT_SPEAKER_WAV if Path(DEFAULT_SPEAKER_WAV).exists() else None
    _voices["default"] = {"id": "default", "name": "Default",
                          "is_cloned": False, "sample_path": default_ref}
    logger.info("loaded %d voices", len(_voices))


def list_voices():
    return list(_voices.values())


def select_voice(requested: str, text: str) -> str:
    """Greek text -> Greek voice; else locked voice; else the requested one."""
    if TTS_VOICE_EL and has_greek(text):
        return TTS_VOICE_EL
    if TTS_LOCK_VOICE:
        return TTS_LOCK_VOICE
    return requested


def resolve_reference(voice: str):
    """Reference wav path for a voice, falling back to the default speaker."""
    v = _voices.get(voice)
    if v and v.get("sample_path") and Path(v["sample_path"]).exists():
        return v["sample_path"]
    if Path(DEFAULT_SPEAKER_WAV).exists():
        return DEFAULT_SPEAKER_WAV
    return None


def save_clone(name: str, audio_bytes: bytes, filename: str) -> str:
    """Store an uploaded sample as a cloned voice (normalized to 16kHz mono wav)."""
    voice_id = f"voice_{hashlib.md5(name.encode()).hexdigest()[:8]}"
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = VOICES_DIR / f"{voice_id}.wav"
    tmp = VOICES_DIR / f"temp_{voice_id}_{Path(filename).name}"
    tmp.write_bytes(audio_bytes)
    try:
        if not filename.lower().endswith(".wav"):
            subprocess.run(["ffmpeg", "-y", "-i", str(tmp), "-ar", "16000", "-ac", "1",
                            str(sample_path)], capture_output=True, check=True)
        else:
            tmp.replace(sample_path)
        _voices[voice_id] = {"id": voice_id, "name": name, "is_cloned": True,
                             "sample_path": str(sample_path), "created_at": time.time()}
        _persist()
        return voice_id
    finally:
        tmp.unlink(missing_ok=True)


def delete_voice(voice_id: str) -> bool:
    """Delete a cloned voice (not presets/default). Returns False if not deletable."""
    v = _voices.get(voice_id)
    if not v or not v.get("is_cloned") or "created_at" not in v:
        return False
    if v.get("sample_path"):
        Path(v["sample_path"]).unlink(missing_ok=True)
    _voices.pop(voice_id, None)
    _persist()
    return True


def _persist():
    """Persist only user-cloned voices; presets/default are reconstructed on load."""
    cloned = {k: v for k, v in _voices.items() if v.get("is_cloned") and "created_at" in v}
    try:
        (VOICES_DIR / "voices.json").write_text(json.dumps(cloned, indent=2))
    except Exception as e:
        logger.error("voices.json save failed: %s", e)
