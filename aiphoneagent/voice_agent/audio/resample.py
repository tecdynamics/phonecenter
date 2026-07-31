"""8 kHz <-> 16 kHz resampling via soxr (VHQ).

Telephony is 8 kHz; ASR/TTS run at 16 kHz. Resampling bugs are a top cause
of garbled audio (spec §11), so keep this one well-tested place for it.
"""

from __future__ import annotations

import numpy as np
import soxr

from . import SAMPLE_RATE_MODEL, SAMPLE_RATE_TELEPHONY


def _resample(pcm: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    if in_rate == out_rate:
        return np.asarray(pcm, dtype=np.int16)
    x = np.asarray(pcm, dtype=np.float32) / 32768.0
    y = soxr.resample(x, in_rate, out_rate, quality="VHQ")
    return np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)


def up_8k_to_16k(pcm8: np.ndarray) -> np.ndarray:
    return _resample(pcm8, SAMPLE_RATE_TELEPHONY, SAMPLE_RATE_MODEL)


def down_16k_to_8k(pcm16: np.ndarray) -> np.ndarray:
    return _resample(pcm16, SAMPLE_RATE_MODEL, SAMPLE_RATE_TELEPHONY)


def resample_to(pcm: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """Generic int16 resample (e.g. 24 kHz XTTS output -> 8 kHz SIP)."""
    return _resample(pcm, in_rate, out_rate)
