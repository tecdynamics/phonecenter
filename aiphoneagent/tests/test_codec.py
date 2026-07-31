"""G.711 codec sanity checks — the one from-scratch algorithm, so verify it.

Run: python -m pytest tests/test_codec.py   (or: python tests/test_codec.py)
"""

from __future__ import annotations

import numpy as np

from voice_agent.audio import codec


def _max_companding_err(pcm: np.ndarray, enc, dec) -> float:
    restored = dec(enc(pcm))
    return float(np.max(np.abs(pcm.astype(np.int32) - restored.astype(np.int32))))


def test_decode_tables_full_range():
    assert codec._ULAW_DECODE.shape == (256,)
    assert codec._ALAW_DECODE.shape == (256,)
    assert codec._ULAW_DECODE.dtype == np.int16
    # Companding is symmetric-ish around zero and spans most of int16.
    assert codec._ULAW_DECODE.max() > 30000
    assert codec._ULAW_DECODE.min() < -30000


def test_ulaw_roundtrip_bounded():
    # Full sweep. At the very top of int16, G.711 clips: u-law's max level is
    # 32124, so 32125..32767 clip (error up to ~643) — that's correct codec
    # behaviour, not a bug. Below the clip region, error stays within a segment.
    pcm = np.arange(-32768, 32768, 7, dtype=np.int16)
    err = _max_companding_err(pcm, codec.ulaw_encode, codec.ulaw_decode)
    assert err <= 700, f"u-law max error too high: {err}"


def test_alaw_roundtrip_bounded():
    pcm = np.arange(-32768, 32768, 7, dtype=np.int16)
    err = _max_companding_err(pcm, codec.alaw_encode, codec.alaw_decode)
    assert err <= 700, f"a-law max error too high: {err}"


def test_small_signals_low_error():
    # Companding is logarithmic: it's most accurate for quiet signals. Near
    # zero the step is tiny (measured: u-law<=8, a-law<=8 over +/-256). a-law
    # has no exact zero level, so silence maps to +/-8 — expected.
    pcm = np.arange(-256, 256, dtype=np.int16)
    assert _max_companding_err(pcm, codec.ulaw_encode, codec.ulaw_decode) <= 8
    assert _max_companding_err(pcm, codec.alaw_encode, codec.alaw_decode) <= 8


def test_payload_type_dispatch():
    pcm = np.array([0, 100, -100, 5000, -5000], dtype=np.int16)
    assert codec.encode(pcm, 0) == codec.ulaw_encode(pcm)
    assert codec.encode(pcm, 8) == codec.alaw_encode(pcm)
    np.testing.assert_array_equal(codec.decode(codec.encode(pcm, 0), 0), codec.ulaw_decode(codec.ulaw_encode(pcm)))


if __name__ == "__main__":
    test_decode_tables_full_range()
    test_ulaw_roundtrip_bounded()
    test_alaw_roundtrip_bounded()
    test_small_signals_low_error()
    test_payload_type_dispatch()
    print("OK: all codec checks passed")
