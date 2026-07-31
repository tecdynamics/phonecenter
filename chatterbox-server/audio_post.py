#!/usr/bin/env python3
"""ffmpeg audio post-processing, matched to the VoxCPM server.

Same env knobs, same filter values as tts-server's convert_audio_format /
_silence_filter, so both engines emit identical-shape audio (8kHz phone,
trimmed silences) for a fair A/B and for drop-in production use.
"""
import os
import uuid
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("chatterbox-pilot")

# Mirror tts-server defaults exactly.
TTS_TRIM_SILENCE = os.environ.get("TTS_TRIM_SILENCE", "true").lower() in ("1", "true", "yes")
TTS_SILENCE_THRESHOLD_DB = os.environ.get("TTS_SILENCE_THRESHOLD_DB", "-40")
TTS_MIN_SILENCE = float(os.environ.get("TTS_MIN_SILENCE", 0.18))
TTS_MAX_GAP = float(os.environ.get("TTS_MAX_GAP", 0.08))
TTS_OUTPUT_SAMPLE_RATE = int(os.environ.get("TTS_OUTPUT_SAMPLE_RATE", 0))
TTS_SPEED = float(os.environ.get("TTS_SPEED", 1.0))  # <1.0 = slower (post-hoc atempo)
MEDIA_DIR = Path(os.environ.get("MEDIA_DIR", "/tec/ai/media/audio"))


def _atempo_chain(speed: float) -> str:
    """ffmpeg atempo chain (a single atempo only spans 0.5-2.0)."""
    factors, r = [], speed
    while r > 2.0:
        factors.append(2.0); r /= 2.0
    while r < 0.5:
        factors.append(0.5); r /= 0.5
    factors.append(r)
    return ",".join(f"atempo={f:.4f}" for f in factors)


def _silence_filter() -> str:
    """Strip leading dead air + cap inter-sentence gaps (identical to VoxCPM)."""
    thr = TTS_SILENCE_THRESHOLD_DB
    return (
        f"silenceremove=start_periods=1:start_silence={TTS_MAX_GAP}:start_threshold={thr}dB:"
        f"stop_periods=-1:stop_duration={TTS_MIN_SILENCE}:"
        f"stop_silence={TTS_MAX_GAP}:stop_threshold={thr}dB"
    )


def postprocess(wav_bytes: bytes, target_format: str = "wav", sample_rate: int = 0,
                speed: float = 1.0) -> bytes:
    """Trim silence, slow/speed, resample, transcode. Returns input on error/no-op."""
    out_rate = int(sample_rate or TTS_OUTPUT_SAMPLE_RATE or 0)
    if not TTS_TRIM_SILENCE and not out_rate and target_format == "wav" and speed == 1.0:
        return wav_bytes

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    src = MEDIA_DIR / f"cbx_{uuid.uuid4().hex[:8]}.wav"
    dst = MEDIA_DIR / f"cbx_{uuid.uuid4().hex[:8]}.{target_format}"
    src.write_bytes(wav_bytes)

    filters = []
    if TTS_TRIM_SILENCE:
        filters.append(_silence_filter())
    if speed != 1.0:
        filters.append(_atempo_chain(speed))
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if filters:
        cmd += ["-af", ",".join(filters)]
    if out_rate:
        cmd += ["-ar", str(out_rate)]
    cmd += ["-q:a", "2", str(dst)]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return dst.read_bytes()
    except Exception as e:
        logger.warning("post-process failed (%s); returning raw wav", e)
        return wav_bytes
    finally:
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)
