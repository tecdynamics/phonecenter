"""Stream-to-clause aggregation.

The LLM streams tokens; XTTS wants speakable chunks. This buffers deltas and
emits a clause as soon as a sentence boundary is seen (so the first clause goes
to TTS fast — low time-to-first-audio), without splitting on decimal points.

Boundaries cover Latin (. ! ? …), Greek (; question mark, · ano teleia) and
Arabic (؟ .). A soft length cap bounds latency if the model omits punctuation.
"""

from __future__ import annotations

# Sentence-ending punctuation across the supported languages.
_BOUNDARY = set(".!?…\n;·؟。！？")


def _is_decimal_dot(buf: str, i: int) -> bool:
    """True if buf[i] is a '.' between two digits (e.g. 19.90) — not a boundary."""
    return (
        buf[i] == "."
        and i > 0
        and buf[i - 1].isdigit()
        and i + 1 < len(buf)
        and buf[i + 1].isdigit()
    )


class ClauseAggregator:
    # Secondary break points used only when a single sentence exceeds max_chars,
    # so an over-long sentence still splits at a natural pause (not mid-phrase).
    _SOFT_BREAKS = ",;:·—"

    def __init__(
        self, min_chars: int = 12, max_chars: int = 200, first_max_chars: int = 0
    ) -> None:
        self._buf = ""
        self._min = min_chars
        self._max = max_chars
        # Time-to-first-audio is set by the FIRST clause alone: synthesis time
        # scales with its length. A tighter cap for that one clause gets the
        # caller hearing something sooner; later clauses use the normal cap and
        # synthesize while the first is already playing. 0 disables.
        self._first_max = first_max_chars
        self._first_done = False

    def _max_now(self) -> int:
        """Cap for the clause being cut right now — tighter until the first emit."""
        if not self._first_done and self._first_max:
            return max(self._first_max, self._min)
        return self._max

    def push(self, text: str) -> list[str]:
        """Append streamed text; return any complete clauses now available."""
        self._buf += text
        clauses: list[str] = []

        # 1. Emit on sentence boundaries.
        emitted_to = 0
        for i, ch in enumerate(self._buf):
            if ch in _BOUNDARY and not _is_decimal_dot(self._buf, i):
                clause = self._buf[emitted_to : i + 1].strip()
                if len(clause) >= self._min:
                    clauses.append(clause)
                    emitted_to = i + 1
        if emitted_to:
            self._buf = self._buf[emitted_to:]

        # 2. If the unterminated remainder is already past the cap, peel it too
        #    (a long run with no sentence end yet).
        while len(self._buf) > self._max_now():
            piece, self._buf = self._peel(self._buf)
            if piece is None:
                break
            clauses.append(piece)

        # 3. Split any over-long clause at natural pauses so XTTS never gets a
        #    chunk beyond its per-utterance limit (that garbles long replies).
        out: list[str] = []
        for c in clauses:
            for piece in self._split_long(c):
                out.append(piece)
                self._first_done = True  # tight cap applied; rest use max_chars
        return out

    def flush(self) -> str:
        """Return whatever is left (call once the stream ends)."""
        tail = self._buf.strip()
        self._buf = ""
        return tail

    def flush_all(self) -> list[str]:
        """Like flush(), but splits an over-long tail into speakable chunks."""
        tail = self.flush()
        return self._split_long(tail) if tail else []

    def _peel(self, s: str) -> tuple[str | None, str]:
        """Cut one <=max chunk off the front of s at a natural pause."""
        window = s[: self._max_now()]
        cut = max(window.rfind(c) for c in self._SOFT_BREAKS)
        if cut < self._min:
            cut = window.rfind(" ")
        if cut <= 0:
            return None, s
        return s[: cut + 1].strip(), s[cut + 1 :].lstrip()

    def _split_long(self, s: str) -> list[str]:
        """Break a clause longer than max_chars into natural-pause sub-chunks."""
        parts: list[str] = []
        while len(s) > self._max_now():
            piece, s = self._peel(s)
            if piece is None:
                break
            parts.append(piece)
        if s.strip():
            parts.append(s.strip())
        return parts
