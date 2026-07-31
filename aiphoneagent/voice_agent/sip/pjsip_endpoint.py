"""pjsua2 (PJSIP) SIP backend — registers the Zadarma extension as a UAC.

Audio is bridged through a custom `AudioMediaPort` running at a 16 kHz bridge
clock, so pjmedia transparently handles G.711 (a-law/u-law) decode/encode and
8k<->16k resampling for us; the port hands the orchestrator clean 16 kHz PCM.

──────────────────────────────────────────────────────────────────────────
⚠️ MUST BE VALIDATED ON THE TARGET BOX. pjsua2 is not pip-installable and is
   absent from the dev machine, so this code cannot be exercised here. Two
   things are genuinely version-dependent across pjsua2 SWIG builds and may
   need adjusting against the installed library:
     1. MediaFrame.buf <-> bytes marshalling (see `_frame_to_bytes` /
        `_bytes_into_frame`).
     2. The exact AudioMediaPort.createPort / format API.
   Validate against phase-1 acceptance: register, answer, greet, echo — one
   clean Greek turn (spec §10).
──────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import threading
from collections.abc import Callable

import numpy as np

from ..audio.aec import EchoCanceller, NullEchoCanceller
from .base import CallHandler, CallSession

logger = logging.getLogger(__name__)

# Factory that returns a FRESH echo canceller per call (adaptive state is
# per-line). main.py builds this from settings; default = no AEC.
AecFactory = Callable[[], EchoCanceller]

try:
    import pjsua2 as pj
except ImportError:  # keeps the package importable on the dev machine
    pj = None  # type: ignore[assignment]

# 16 kHz / mono / 16-bit / 20 ms bridge frames.
_CLOCK_RATE = 16000
_CHANNELS = 1
_BITS = 16
_FRAME_USEC = 20000
_FRAME_BYTES = _CLOCK_RATE * _CHANNELS * (_BITS // 8) * _FRAME_USEC // 1_000_000  # 640


_SIP_USER_RE = re.compile(r"sip:([^@;>]+)@")


def _parse_caller_id(remote_uri: str) -> str:
    """Extract the user part (caller number) from a SIP URI like '"X" <sip:NUM@host>'."""
    m = _SIP_USER_RE.search(remote_uri or "")
    return m.group(1).strip() if m else ""


def _require_pj() -> None:
    if pj is None:
        raise RuntimeError(
            "pjsua2 is not installed. Build PJSIP with the Python bindings on the server "
            "(see README §SIP). It is intentionally absent from the dev environment."
        )


def _frame_to_bytes(frame) -> bytes:
    """Extract PCM bytes from a pjsua2 MediaFrame. See module caveat (1)."""
    buf = frame.buf
    try:
        return bytes(buf)
    except TypeError:
        return bytes(bytearray(buf[i] for i in range(frame.size)))


def _bytes_into_frame(frame, data: bytes) -> None:
    """Fill a MediaFrame's buffer with PCM bytes. See module caveat (1)."""
    frame.type = pj.PJMEDIA_FRAME_TYPE_AUDIO
    try:
        frame.buf = pj.ByteVector(data)
    except (TypeError, AttributeError):
        vec = frame.buf
        try:
            vec.clear()
        except AttributeError:
            pass
        for b in data:
            vec.append(b)
    frame.size = len(data)


class _PcmBridgePort:
    """Custom AudioMediaPort: caller audio in, agent audio out, both 16 kHz PCM.

    Defined lazily (subclasses pj.AudioMediaPort) so the module imports without
    pjsua2 present.
    """

    def __new__(cls, *args, **kwargs):  # pragma: no cover - requires pjsua2
        _require_pj()
        return _build_bridge_port_class()(*args, **kwargs)


_BRIDGE_PORT_CLS = None


def _build_bridge_port_class():  # pragma: no cover - requires pjsua2
    global _BRIDGE_PORT_CLS
    if _BRIDGE_PORT_CLS is not None:
        return _BRIDGE_PORT_CLS

    class BridgePort(pj.AudioMediaPort):
        def __init__(self, session: "_PjCallSession", loop: asyncio.AbstractEventLoop):
            super().__init__()
            self._session = session
            self._loop = loop
            fmt = pj.MediaFormatAudio()
            fmt.type = pj.PJMEDIA_TYPE_AUDIO
            fmt.clockRate = _CLOCK_RATE
            fmt.channelCount = _CHANNELS
            fmt.bitsPerSample = _BITS
            fmt.frameTimeUsec = _FRAME_USEC
            self.createPort(f"agent-{session.call_id}", fmt)

        def onFrameReceived(self, frame):
            # Runs on a pjmedia thread (already pj-registered). Marshal to loop.
            if frame.size <= 0:
                return
            # Cancel the agent's own audio echoing back before anything sees it,
            # so Whisper/VAD work on clean caller audio (buffered → may be empty).
            clean = self._session._aec.process_captured(_frame_to_bytes(frame))
            if not clean:
                return
            pcm = np.frombuffer(clean, dtype=np.int16).copy()
            self._loop.call_soon_threadsafe(self._session.inbound.put_nowait, pcm)

        def onFrameRequested(self, frame):
            out = self._session._pop_output(_FRAME_BYTES)
            # Record exactly what we play as the AEC far-end reference.
            self._session._aec.notify_played(out)
            _bytes_into_frame(frame, out)

    _BRIDGE_PORT_CLS = BridgePort
    return BridgePort


class _PjCallSession(CallSession):
    def __init__(
        self,
        call_id: str,
        call,
        loop: asyncio.AbstractEventLoop,
        sip_server: str = "",
        aec: EchoCanceller | None = None,
    ) -> None:
        super().__init__(call_id)
        self._call = call
        self._loop = loop
        self._sip_server = sip_server
        self._out = bytearray()
        self._out_lock = threading.Lock()
        # Per-call echo canceller; read/written on the pjmedia thread only.
        self._aec: EchoCanceller = aec or NullEchoCanceller()
        self.port = None  # set after media is up

    # ── outbound buffer (written from asyncio, read from media thread) ──
    async def send_audio(self, pcm16_16k: np.ndarray) -> None:
        data = np.asarray(pcm16_16k, dtype=np.int16).tobytes()
        with self._out_lock:
            self._out.extend(data)

    def flush_output(self) -> None:
        with self._out_lock:
            self._out.clear()

    def _pop_output(self, n: int) -> bytes:
        with self._out_lock:
            if len(self._out) >= n:
                chunk = bytes(self._out[:n])
                del self._out[:n]
                return chunk
            chunk = bytes(self._out)
            self._out.clear()
        return chunk + b"\x00" * (n - len(chunk))  # pad with silence

    @property
    def output_pending(self) -> bool:
        with self._out_lock:
            return len(self._out) > 0

    async def transfer(self, extension: str) -> None:  # pragma: no cover - requires pjsua2
        _require_pj()
        # Blind transfer (SIP REFER). Bare extensions are sent as sip:<ext>@<server>.
        target = extension if extension.startswith("sip:") else f"sip:{extension}@{self._sip_server}"
        logger.info("Call %s: REFER transfer to %s", self.call_id, target)
        self._call.xfer(target, pj.CallOpParam())

    async def hangup(self) -> None:  # pragma: no cover - requires pjsua2
        _require_pj()
        prm = pj.CallOpParam()
        prm.statusCode = pj.PJSIP_SC_OK
        try:
            self._call.hangup(prm)
        except Exception:  # noqa: BLE001 - call may already be gone
            logger.debug("hangup on already-terminated call %s", self.call_id)

    def _on_ended(self) -> None:
        self._ended.set()
        self.inbound.put_nowait(None)


class Pjsua2Transport:  # implements base.SipTransport
    """pjsua2 endpoint + account that REGISTERs the extension and answers calls."""

    def __init__(
        self,
        on_call: CallHandler,
        *,
        sip_server: str,
        extension: str,
        password: str,
        transport: str = "udp",
        max_concurrent_calls: int = 2,
        aec_factory: AecFactory | None = None,
    ) -> None:
        _require_pj()
        self._on_call = on_call
        self._sip_server = sip_server
        self._extension = extension
        self._password = password
        self._transport = transport
        self._max_calls = max_concurrent_calls
        # Builds a fresh per-call echo canceller; None → no AEC (NullEchoCanceller).
        self._aec_factory = aec_factory
        self._ep = None
        self._account = None
        self._active = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:  # pragma: no cover - requires pjsua2
        self._loop = asyncio.get_running_loop()
        await asyncio.to_thread(self._start_blocking)
        # transfer()/hangup() call pjlib from THIS (asyncio loop) thread. pjsua2
        # aborts the process if an unregistered external thread calls pjlib, so
        # register the loop thread now (it persists for the service lifetime).
        if self._ep is not None:
            with contextlib.suppress(Exception):
                self._ep.libRegisterThread("asyncio-loop")

    def _start_blocking(self) -> None:  # pragma: no cover - requires pjsua2
        ep = pj.Endpoint()
        ep.libCreate()

        ep_cfg = pj.EpConfig()
        ep_cfg.medConfig.clockRate = _CLOCK_RATE
        ep_cfg.medConfig.channelCount = _CHANNELS
        ep_cfg.medConfig.ptime = _FRAME_USEC // 1000
        ep_cfg.uaConfig.maxCalls = max(self._max_calls, 1)
        ep.libInit(ep_cfg)

        tp_type = {
            "udp": pj.PJSIP_TRANSPORT_UDP,
            "tcp": pj.PJSIP_TRANSPORT_TCP,
            "tls": pj.PJSIP_TRANSPORT_TLS,
        }[self._transport]
        ep.transportCreate(tp_type, pj.TransportConfig())
        ep.libStart()
        # No sound device — we bridge audio ourselves via the custom port.
        ep.audDevManager().setNullDev()

        acc_cfg = pj.AccountConfig()
        acc_cfg.idUri = f"sip:{self._extension}@{self._sip_server}"
        acc_cfg.regConfig.registrarUri = f"sip:{self._sip_server}"
        cred = pj.AuthCredInfo("digest", "*", self._extension, 0, self._password)
        acc_cfg.sipConfig.authCreds.append(cred)
        # pjsua2 auto re-registers; keep the default refresh + retry-on-failure.

        account = _build_account_class()(self)
        account.create(acc_cfg)

        self._ep = ep
        self._account = account
        logger.info("pjsua2 started; REGISTER sent for extension %s", self._extension)

    async def stop(self) -> None:  # pragma: no cover - requires pjsua2
        await asyncio.to_thread(self._stop_blocking)

    def _stop_blocking(self) -> None:  # pragma: no cover - requires pjsua2
        if self._ep is not None:
            try:
                self._ep.libDestroy()
            finally:
                self._ep = None
                self._account = None

    # Called from the account callback (pj thread) when a call connects.
    def _dispatch_call(self, session: CallSession) -> None:  # pragma: no cover
        assert self._loop is not None
        asyncio.run_coroutine_threadsafe(self._on_call(session), self._loop)


_ACCOUNT_CLS = None


def _build_account_class():  # pragma: no cover - requires pjsua2
    global _ACCOUNT_CLS
    if _ACCOUNT_CLS is not None:
        return _ACCOUNT_CLS

    class _Account(pj.Account):
        def __init__(self, transport: Pjsua2Transport):
            super().__init__()
            self._transport = transport
            self._calls: dict[int, object] = {}

        def onRegState(self, prm):
            logger.info("Registration state: code=%s", prm.code)

        def onIncomingCall(self, prm):
            t = self._transport
            call = _build_call_class()(self, t, prm.callId)
            answer = pj.CallOpParam()
            if t._active >= t._max_calls:
                answer.statusCode = pj.PJSIP_SC_BUSY_HERE
                call.answer(answer)
                logger.warning("Rejected call %s: at max concurrency", prm.callId)
                return
            t._active += 1
            self._calls[prm.callId] = call
            answer.statusCode = pj.PJSIP_SC_OK  # 200: answer immediately
            call.answer(answer)

    _ACCOUNT_CLS = _Account
    return _Account


_CALL_CLS = None


def _build_call_class():  # pragma: no cover - requires pjsua2
    global _CALL_CLS
    if _CALL_CLS is not None:
        return _CALL_CLS

    class _Call(pj.Call):
        def __init__(self, account, transport: Pjsua2Transport, call_id: int):
            super().__init__(account, call_id)
            self._transport = transport
            self._session: _PjCallSession | None = None
            self._port = None

        def onCallState(self, prm):
            info = self.getInfo()
            if info.state == pj.PJSIP_INV_STATE_DISCONNECTED:
                t = self._transport
                t._active = max(0, t._active - 1)
                if self._session is not None and t._loop is not None:
                    t._loop.call_soon_threadsafe(self._session._on_ended)
                logger.info("Call %s disconnected", info.id)

        def onCallMediaState(self, prm):
            info = self.getInfo()
            for i, mi in enumerate(info.media):
                if mi.type == pj.PJMEDIA_TYPE_AUDIO and mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                    call_audio = self.getAudioMedia(i)
                    t = self._transport
                    aec = t._aec_factory() if t._aec_factory is not None else None
                    session = _PjCallSession(str(info.id), self, t._loop, t._sip_server, aec)
                    session.caller_id = _parse_caller_id(info.remoteUri)
                    # localUri is the called party. With a cloud PBX fanning several
                    # DIDs into ONE registered extension it may be rewritten to that
                    # extension and be identical for every number — log both so
                    # per-DID routing can be built on whatever field actually varies.
                    session.dialed_number = _parse_caller_id(info.localUri)
                    logger.info("Call %s from %s | dialed=%r localUri=%s",
                                info.id, info.remoteUri, session.dialed_number, info.localUri)
                    port = _build_bridge_port_class()(session, t._loop)
                    session.port = port
                    # remote -> port (capture) and port -> remote (playback)
                    call_audio.startTransmit(port)
                    port.startTransmit(call_audio)
                    self._session = session
                    self._port = port
                    t._dispatch_call(session)
                    break

        def onDtmfDigit(self, prm):
            t = self._transport
            if self._session is not None and t._loop is not None:
                t._loop.call_soon_threadsafe(self._session.dtmf.put_nowait, prm.digit)

    _CALL_CLS = _Call
    return _Call
