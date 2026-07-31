"""SIP transport abstractions — the seam that keeps pjsua2 swappable.

The orchestrator depends only on these. A second backend (e.g. self-hosted
LiveKit SIP) just needs to implement `SipTransport` + `CallSession`.

Audio crossing this boundary is always **mono int16 PCM at 16 kHz**. G.711
companding and 8k<->16k resampling live below this line (the pjsua2 backend
lets pjmedia do it at the bridge clock rate; see pjsip_endpoint.py).
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Awaitable, Callable

import numpy as np


class CallSession(abc.ABC):
    """One active inbound call. Handed to the orchestrator on connect."""

    def __init__(self, call_id: str) -> None:
        self.call_id = call_id
        # Caller's number from the SIP From header (CLI), if Zadarma provides it.
        self.caller_id: str = ""
        # Number the caller DIALLED (the DID), from the SIP local/To URI. Used to
        # pick the greeting language per number. May be empty, or identical for
        # every DID, if the PBX routes them all into one registered extension.
        self.dialed_number: str = ""
        # Inbound 16 kHz PCM chunks (producer: SIP media thread).
        self.inbound: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        # DTMF digits surfaced for menu/verification flows (phase 4).
        self.dtmf: asyncio.Queue[str] = asyncio.Queue()
        self._ended = asyncio.Event()

    @abc.abstractmethod
    async def send_audio(self, pcm16_16k: np.ndarray) -> None:
        """Queue 16 kHz PCM for playback to the caller."""

    @abc.abstractmethod
    def flush_output(self) -> None:
        """Drop any buffered outbound audio immediately (barge-in, phase 2)."""

    @property
    @abc.abstractmethod
    def output_pending(self) -> bool:
        """True while buffered outbound audio is still being played out."""

    @abc.abstractmethod
    async def transfer(self, extension: str) -> None:
        """Blind transfer to a human (SIP REFER, phase 4)."""

    @abc.abstractmethod
    async def hangup(self) -> None:
        """End the call and release resources."""

    @property
    def ended(self) -> asyncio.Event:
        return self._ended


# Called by the transport when a call connects; receives the live session.
CallHandler = Callable[[CallSession], Awaitable[None]]


class SipTransport(abc.ABC):
    """Registers the extension and dispatches inbound calls to a handler."""

    def __init__(self, on_call: CallHandler) -> None:
        self._on_call = on_call

    @abc.abstractmethod
    async def start(self) -> None:
        """Init the stack, REGISTER, and begin accepting calls."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Unregister and tear down the stack."""
