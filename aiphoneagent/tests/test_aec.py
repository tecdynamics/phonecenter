"""AEC wiring + frame-buffering checks.

The real speexdsp lib is server-only, so the adaptive-filter quality can only be
judged on a live call. What we CAN verify on the dev box is the buffering /
far-near pairing logic and the graceful fallback — done here by injecting a fake
`speexdsp` whose `process(near, far)` is a trivial subtraction.

Run: python -m pytest tests/test_aec.py   (or: python tests/test_aec.py)
"""

from __future__ import annotations

import sys
import types

import numpy as np

from voice_agent.audio import aec


def _i16(*vals: int) -> bytes:
    return np.array(vals, dtype=np.int16).tobytes()


class _FakeSpeexEC:
    """Stand-in for speexdsp.EchoCanceller: cleaned = near - far (per frame)."""

    @classmethod
    def create(cls, frame_size, filter_length, sample_rate):
        inst = cls()
        inst.frame_size = frame_size
        return inst

    def process(self, near_bytes: bytes, far_bytes: bytes) -> bytes:
        near = np.frombuffer(near_bytes, dtype=np.int16).astype(np.int32)
        far = np.frombuffer(far_bytes, dtype=np.int16).astype(np.int32)
        return (near - far).astype(np.int16).tobytes()


def _install_fake_speex() -> None:
    mod = types.ModuleType("speexdsp")
    mod.EchoCanceller = _FakeSpeexEC
    sys.modules["speexdsp"] = mod


def test_null_canceller_passes_through():
    ec = aec.NullEchoCanceller()
    ec.notify_played(_i16(1, 2, 3))  # no-op
    assert ec.process_captured(_i16(4, 5, 6)) == _i16(4, 5, 6)


def test_disabled_builds_null():
    ec = aec.build_echo_canceller(enabled=False, frame_size=320, tail_ms=200, sample_rate=16000)
    assert isinstance(ec, aec.NullEchoCanceller)


def test_missing_speexdsp_falls_back_to_null():
    # Ensure no speexdsp is importable, then request AEC → must degrade, not raise.
    saved = sys.modules.pop("speexdsp", None)
    try:
        sys.modules["speexdsp"] = None  # forces ImportError on `import speexdsp`
        ec = aec.build_echo_canceller(enabled=True, frame_size=320, tail_ms=200, sample_rate=16000)
        assert isinstance(ec, aec.NullEchoCanceller)
    finally:
        if saved is not None:
            sys.modules["speexdsp"] = saved
        else:
            sys.modules.pop("speexdsp", None)


def test_subtracts_paired_far_frame():
    _install_fake_speex()
    ec = aec.SpeexEchoCanceller(frame_size=2, filter_length=4, sample_rate=16000)
    ec.notify_played(_i16(10, 10))           # far frame
    out = ec.process_captured(_i16(30, 25))  # near frame → near - far
    assert np.array_equal(np.frombuffer(out, dtype=np.int16), np.array([20, 15], dtype=np.int16))


def test_silence_reference_when_nothing_played():
    _install_fake_speex()
    ec = aec.SpeexEchoCanceller(frame_size=2, filter_length=4, sample_rate=16000)
    # No notify_played → far is silence → near passes through unchanged.
    out = ec.process_captured(_i16(7, 9))
    assert np.array_equal(np.frombuffer(out, dtype=np.int16), np.array([7, 9], dtype=np.int16))


def test_buffers_partial_frame():
    _install_fake_speex()
    ec = aec.SpeexEchoCanceller(frame_size=2, filter_length=4, sample_rate=16000)
    # Only one sample (< one 2-sample frame) → nothing emitted yet.
    assert ec.process_captured(_i16(5)) == b""
    # Second sample completes the frame → emitted (no far → unchanged).
    out = ec.process_captured(_i16(6))
    assert np.array_equal(np.frombuffer(out, dtype=np.int16), np.array([5, 6], dtype=np.int16))


def test_far_backlog_is_capped():
    _install_fake_speex()
    ec = aec.SpeexEchoCanceller(frame_size=2, filter_length=4, sample_rate=16000)
    # Flood far with far more than the cap; it must not grow unbounded.
    for _ in range(10_000):
        ec.notify_played(_i16(1, 1))
    assert len(ec._far) <= ec._far_cap


def test_reset_clears_buffers():
    _install_fake_speex()
    ec = aec.SpeexEchoCanceller(frame_size=2, filter_length=4, sample_rate=16000)
    ec.notify_played(_i16(1, 1))
    ec.process_captured(_i16(9))  # leaves a partial near sample buffered
    ec.reset()
    assert len(ec._far) == 0 and len(ec._near) == 0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
