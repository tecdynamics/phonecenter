"""G.711 a-law / u-law (PCMU/PCMA) codec — pure numpy.

`audioop` was removed in Python 3.13, so we build the companding tables
ourselves. Decode tables come straight from the ITU G.711 definitions;
encode tables are the nearest-decoded-value inverse (the standard segment
thresholds are midpoints, so nearest-value reproduces the codec exactly).

All functions are vectorized: bytes <-> int16 numpy arrays.
"""

from __future__ import annotations

import numpy as np

_BIAS = 0x84  # 132
_CLIP = 32635


def _build_ulaw_decode() -> np.ndarray:
    out = np.empty(256, dtype=np.int16)
    for code in range(256):
        u = ~code & 0xFF
        sign = u & 0x80
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        magnitude = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
        out[code] = -magnitude if sign else magnitude
    return out


def _build_alaw_decode() -> np.ndarray:
    out = np.empty(256, dtype=np.int16)
    for code in range(256):
        a = code ^ 0x55
        sign = a & 0x80
        exponent = (a >> 4) & 0x07
        mantissa = a & 0x0F
        if exponent == 0:
            magnitude = (mantissa << 4) + 8
        else:
            magnitude = ((mantissa << 4) + 0x108) << (exponent - 1)
        out[code] = -magnitude if sign else magnitude
    return out


def _build_encode_table(decode: np.ndarray) -> np.ndarray:
    """uint8 encode LUT indexed by (int16 sample + 32768): nearest decoded code."""
    samples = np.arange(-32768, 32768, dtype=np.int32)
    order = np.argsort(decode, kind="stable")
    dec_sorted = decode[order].astype(np.int32)

    pos = np.searchsorted(dec_sorted, samples)
    pos_lo = np.clip(pos - 1, 0, len(dec_sorted) - 1)
    pos_hi = np.clip(pos, 0, len(dec_sorted) - 1)
    d_lo = np.abs(samples - dec_sorted[pos_lo])
    d_hi = np.abs(samples - dec_sorted[pos_hi])
    chosen = np.where(d_lo <= d_hi, pos_lo, pos_hi)
    return order[chosen].astype(np.uint8)


_ULAW_DECODE = _build_ulaw_decode()
_ALAW_DECODE = _build_alaw_decode()
_ULAW_ENCODE = _build_encode_table(_ULAW_DECODE)
_ALAW_ENCODE = _build_encode_table(_ALAW_DECODE)


def ulaw_decode(data: bytes) -> np.ndarray:
    return _ULAW_DECODE[np.frombuffer(data, dtype=np.uint8)]


def alaw_decode(data: bytes) -> np.ndarray:
    return _ALAW_DECODE[np.frombuffer(data, dtype=np.uint8)]


def ulaw_encode(pcm: np.ndarray) -> bytes:
    return _ULAW_ENCODE[np.asarray(pcm, dtype=np.int16).astype(np.int32) + 32768].tobytes()


def alaw_encode(pcm: np.ndarray) -> bytes:
    return _ALAW_ENCODE[np.asarray(pcm, dtype=np.int16).astype(np.int32) + 32768].tobytes()


def decode(data: bytes, payload_type: int) -> np.ndarray:
    """Decode G.711 by RTP payload type (0=PCMU, 8=PCMA) to int16 PCM."""
    if payload_type == 0:
        return ulaw_decode(data)
    if payload_type == 8:
        return alaw_decode(data)
    raise ValueError(f"Unsupported G.711 payload type: {payload_type}")


def encode(pcm: np.ndarray, payload_type: int) -> bytes:
    if payload_type == 0:
        return ulaw_encode(pcm)
    if payload_type == 8:
        return alaw_encode(pcm)
    raise ValueError(f"Unsupported G.711 payload type: {payload_type}")
