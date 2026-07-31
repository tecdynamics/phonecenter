"""Per-stage latency + per-call structured logging.

Spec §6 sets a latency budget per turn (VAD, ASR, RAG, LLM-TTFT, TTS-TTFA);
measure each stage so regressions are visible. Phase 1 records ASR and TTS.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger("voice_agent.metrics")


@dataclass
class TurnMetrics:
    call_id: str
    stages_ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages_ms[name] = round((time.perf_counter() - start) * 1000, 1)

    def log(self, **extra: object) -> None:
        total = round(sum(self.stages_ms.values()), 1)
        logger.info(
            "turn call=%s total_ms=%s stages=%s %s",
            self.call_id,
            total,
            self.stages_ms,
            " ".join(f"{k}={v}" for k, v in extra.items()),
        )
