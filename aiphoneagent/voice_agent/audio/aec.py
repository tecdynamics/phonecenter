"""Acoustic echo cancellation for the SIP bridge.

The agent plays its TTS out to the caller and that audio echoes back on the
line (no hardware AEC — Zadarma is a cloud PBX and we bridge through a custom
pjsua2 `AudioMediaPort`). Whisper then transcribes the agent's own voice as a
phantom caller turn, and the barge-in VAD flushes the agent's own reply. That
is why barge-in was disabled (see `Settings.bargein_enabled`).

This module cancels that echo at the one place that has BOTH signals:
  • the **far-end reference** — exactly the PCM we hand to pjmedia for playback
    (`_PcmBridgePort.onFrameRequested` → `CallSession._pop_output`), and
  • the **near-end capture** — the frame pjmedia gives us from the caller
    (`onFrameReceived`).

`notify_played()` records the reference; `process_captured()` subtracts the
echo and returns clean near-end PCM. Both run on the pjmedia thread, frame by
frame, so the canceller buffers internally to its fixed frame size and pairs
one far-end frame per near-end frame. The far/near pairing is only frame-clock
aligned (~20 ms); the adaptive filter's tail (`tail_ms`) absorbs the real
round-trip echo delay, so size the tail to the worst-case line delay.

Behind a small `EchoCanceller` protocol so the implementation is swappable and
so the package imports with no speexdsp present (dev box) — `build_echo_cancel`
falls back to a transparent no-op and logs once.

⚠️ Like the rest of `sip/`, this can only be exercised on the server (needs a
real call + speexdsp). Validate with a live call: speak over the agent and
confirm Whisper no longer transcribes the agent's own audio.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class EchoCanceller(Protocol):
    """Frame-streaming echo canceller. PCM is mono int16 `bytes` throughout."""

    def notify_played(self, far: bytes) -> None:
        """Record a chunk of far-end (played-to-caller) audio as the reference."""

    def process_captured(self, near: bytes) -> bytes:
        """Return near-end (captured) audio with the recorded echo removed.

        May return fewer/more bytes than passed in (internal frame buffering);
        returns `b""` until at least one full frame is available.
        """

    def reset(self) -> None:
        """Drop buffered audio and adaptive state (call at the start of a call)."""


class NullEchoCanceller:
    """Transparent pass-through — used when AEC is off or speexdsp is absent."""

    def notify_played(self, far: bytes) -> None:  # noqa: D102
        return None

    def process_captured(self, near: bytes) -> bytes:  # noqa: D102
        return near

    def reset(self) -> None:  # noqa: D102
        return None


class SpeexEchoCanceller:
    """speexdsp-backed AEC, buffered to a fixed frame size.

    `frame_size`/`filter_length` are in samples (int16). speex wants the filter
    length a few times the frame size and comfortably longer than the echo
    round-trip; `filter_length = tail_ms * sample_rate / 1000`.
    """

    def __init__(self, frame_size: int, filter_length: int, sample_rate: int) -> None:
        # Lazy import: libspeexdsp is server-only and absent on the dev box.
        from speexdsp import EchoCanceller as _SpeexEC

        self._frame_size = frame_size
        self._frame_bytes = frame_size * 2  # int16
        self._sample_rate = sample_rate
        self._filter_length = filter_length
        self._make = lambda: _SpeexEC.create(frame_size, filter_length, sample_rate)
        self._ec = self._make()
        # Far-end frames played but not yet paired with a captured frame, and the
        # leftover near-end bytes shorter than one full frame.
        self._far = bytearray()
        self._near = bytearray()
        # Bound the far backlog so a callback-rate drift can't grow it forever
        # (worst case the filter just sees a small extra delay).
        self._far_cap = self._frame_bytes * max(8, filter_length // frame_size + 4)

    def notify_played(self, far: bytes) -> None:
        self._far.extend(far)
        if len(self._far) > self._far_cap:
            del self._far[: len(self._far) - self._far_cap]

    def process_captured(self, near: bytes) -> bytes:
        self._near.extend(near)
        out = bytearray()
        while len(self._near) >= self._frame_bytes:
            near_frame = bytes(self._near[: self._frame_bytes])
            del self._near[: self._frame_bytes]
            if len(self._far) >= self._frame_bytes:
                far_frame = bytes(self._far[: self._frame_bytes])
                del self._far[: self._frame_bytes]
            else:
                # Nothing being played → no echo to cancel; reference is silence.
                far_frame = self._frame_bytes * b"\x00"
            out.extend(self._ec.process(near_frame, far_frame))
        return bytes(out)

    def reset(self) -> None:
        self._far.clear()
        self._near.clear()
        # Fresh adaptive filter per call (echo path differs per caller/line).
        self._ec = self._make()


def build_echo_canceller(
    *,
    enabled: bool,
    frame_size: int,
    tail_ms: int,
    sample_rate: int,
) -> EchoCanceller:
    """Construct a per-call canceller; fall back to a no-op on any problem.

    Returns a transparent `NullEchoCanceller` when AEC is disabled or speexdsp
    is unavailable, so the agent always runs — it just keeps the echo.
    """
    if not enabled:
        return NullEchoCanceller()
    filter_length = max(frame_size, sample_rate * tail_ms // 1000)
    try:
        ec = SpeexEchoCanceller(frame_size, filter_length, sample_rate)
        logger.info(
            "AEC on (speexdsp): frame=%d filter=%d (%d ms) @ %d Hz",
            frame_size, filter_length, tail_ms, sample_rate,
        )
        return ec
    except Exception:  # noqa: BLE001 - missing lib / build mismatch → degrade, don't crash
        logger.warning(
            "AEC requested but speexdsp is unavailable — running WITHOUT echo "
            "cancellation (install `speexdsp`). Barge-in should stay off.",
            exc_info=True,
        )
        return NullEchoCanceller()


def pcm_bytes(pcm: np.ndarray) -> bytes:
    """int16 ndarray → bytes (helper for callers holding numpy frames)."""
    return np.asarray(pcm, dtype=np.int16).tobytes()
