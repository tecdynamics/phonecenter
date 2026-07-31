"""VAD-based endpointing.

Behind a small `Endpointer` protocol so the implementation is swappable
(phase 2 may switch to Pipecat's built-in Silero processor). Phase 1 uses
the standalone `silero-vad` package directly.

Feed 16 kHz int16 PCM chunks via `process()`; it returns an event and, on
end-of-utterance, the buffered utterance audio (with a short pre-roll so the
onset isn't clipped).
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol

import numpy as np

from . import SAMPLE_RATE_MODEL


class VadEvent(Enum):
    NONE = "none"
    SPEECH_START = "speech_start"
    UTTERANCE_END = "utterance_end"


class Endpointer(Protocol):
    def reset(self) -> None: ...
    def process(self, pcm16: np.ndarray) -> tuple[VadEvent, np.ndarray | None]: ...


class SileroEndpointer:
    """Streaming end-of-utterance detector built on silero-vad."""

    WINDOW = 512  # samples @16 kHz (~32 ms), the silero frame size
    PREROLL_MS = 200

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 200,
        silence_ms: int = 500,
    ) -> None:
        # Lazy imports: torch/silero only needed when VAD actually runs.
        import torch  # noqa: F401
        from silero_vad import load_silero_vad

        self._torch = torch
        self._model = load_silero_vad()
        self._threshold = threshold
        self._min_speech_samples = min_speech_ms * SAMPLE_RATE_MODEL // 1000
        self._silence_samples = silence_ms * SAMPLE_RATE_MODEL // 1000
        self._preroll_samples = self.PREROLL_MS * SAMPLE_RATE_MODEL // 1000
        self.reset()

    def reset(self) -> None:
        self._model.reset_states()
        self._leftover = np.empty(0, dtype=np.int16)
        self._preroll = np.empty(0, dtype=np.int16)
        self._utterance: list[np.ndarray] = []
        self._triggered = False
        self._speech_run = 0
        self._silence_run = 0

    def _speech_prob(self, window: np.ndarray) -> float:
        t = self._torch.from_numpy(window.astype(np.float32) / 32768.0)
        with self._torch.no_grad():
            return float(self._model(t, SAMPLE_RATE_MODEL).item())

    def process(self, pcm16: np.ndarray) -> tuple[VadEvent, np.ndarray | None]:
        buf = np.concatenate([self._leftover, np.asarray(pcm16, dtype=np.int16)])
        n_full = len(buf) // self.WINDOW
        self._leftover = buf[n_full * self.WINDOW :]

        event = VadEvent.NONE
        for i in range(n_full):
            window = buf[i * self.WINDOW : (i + 1) * self.WINDOW]
            is_speech = self._speech_prob(window) >= self._threshold

            if not self._triggered:
                # Keep a rolling pre-roll so we don't clip the onset.
                self._preroll = np.concatenate([self._preroll, window])[-self._preroll_samples :]
                if is_speech:
                    self._speech_run += self.WINDOW
                    if self._speech_run >= self._min_speech_samples:
                        self._triggered = True
                        self._silence_run = 0
                        self._utterance = [self._preroll.copy()]
                        event = VadEvent.SPEECH_START
                else:
                    self._speech_run = 0
            else:
                self._utterance.append(window)
                if is_speech:
                    self._silence_run = 0
                else:
                    self._silence_run += self.WINDOW
                    if self._silence_run >= self._silence_samples:
                        audio = np.concatenate(self._utterance)
                        self._triggered = False
                        self._speech_run = 0
                        self._utterance = []
                        self._preroll = np.empty(0, dtype=np.int16)
                        return VadEvent.UTTERANCE_END, audio
        return event, None
