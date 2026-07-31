"""Call orchestrators.

`EchoOrchestrator` is the phase-1 loop (answer -> greeting -> ASR -> echo),
kept for diagnostics. `ConversationOrchestrator` is the phase-2 loop:

    greeting -> listen -> ASR -> Gemma (streaming) -> XTTS, with barge-in.

Barge-in is done without Pipecat: while the agent speaks, VAD runs on inbound
audio; on detected caller speech we flush the outbound buffer
(`CallSession.flush_output`), cancel the in-flight generation/synthesis, and
capture the caller's utterance so the next turn uses it. RAG is phase 3.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import AsyncIterator

import numpy as np

from .asr.whisper_asr import WhisperASR
from .audio.vad import SileroEndpointer, VadEvent
from .config import Settings
from .llm.clause import ClauseAggregator
from .llm.conversation import Conversation
from .llm.gemma_client import GemmaClient, LLMError
from .metrics import TurnMetrics
from .notify.mailer import Mailer
from .rag.retriever import Retriever
from .sip.base import CallSession
from .tts.xtts_client import XttsClient

logger = logging.getLogger(__name__)

# Control tokens the LLM emits (never spoken; stripped before TTS).
# [[TRANSFER]] or [[TRANSFER:support]] → hand off; [[HANGUP]] → end the call.
_TRANSFER_RE = re.compile(r"\[\[TRANSFER(?::\s*([A-Za-z_]+))?\]\]")
_HANGUP_MARKER = "[[HANGUP]]"
_MARKER_RE = re.compile(r"\[\[[^\]]*\]\]")

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _detect_contacts(transcript: str) -> tuple[str | None, str | None]:
    """Best-effort pull of an email and the longest phone-like digit run."""
    email = _EMAIL_RE.search(transcript)
    runs = re.findall(r"[\d ]{6,}", transcript)
    digits = max((re.sub(r"\D", "", r) for r in runs), key=len, default="")
    return (email.group(0) if email else None, digits or None)

# Spoken phrases for the echo demo, per supported language.
_GREETING = {
    "el": "Καλώς ήρθατε. Πείτε μου κάτι και θα το επαναλάβω.",
    "en": "Hello. Say something and I will repeat it back.",
    "ar": "مرحبا. قل شيئا وسوف أكرره.",
}
_ECHO_PREFIX = {"el": "Είπατε:", "en": "You said:", "ar": "لقد قلت:"}
_STILL_THERE = {
    "el": "Είστε ακόμα εκεί;",
    "en": "Are you still there?",
    "ar": "هل ما زلت هناك؟",
}

# No-input handling.
_NO_INPUT_TIMEOUT_S = 8.0
# Cap on how long we keep collecting a barge-in utterance before giving up.
_UTTERANCE_MAX_S = 15.0

# Spoken when a transfer was attempted but the call didn't hand off (PBX rejected
# REFER) — apologize and promise a callback instead of leaving the caller in silence.
_TRANSFER_FAILED = {
    "el": "Συγγνώμη, δεν μπόρεσα να σας συνδέσω αυτή τη στιγμή. Θα ζητήσω από έναν συνάδελφο να σας καλέσει σύντομα.",
    "en": "I'm sorry, I couldn't connect you right now. I'll have a colleague call you back as soon as possible.",
    "ar": "آسف، لم أتمكن من توصيلك الآن. سأطلب من زميل أن يعاود الاتصال بك قريبًا.",
}

# Graceful degradation when the LLM is unreachable (spec §9: apologize + human).
_LLM_FAIL = {
    "el": "Συγγνώμη, αντιμετωπίζω μια τεχνική δυσκολία αυτή τη στιγμή. Θα σας συνδέσω με έναν συνάδελφο.",
    "en": "I'm sorry, I'm having a technical problem right now. Let me connect you with a colleague.",
    "ar": "آسف، أواجه مشكلة تقنية الآن. دعني أحوّلك إلى أحد الزملاء.",
}

# Short acknowledgement spoken instantly while the real reply generates (masks latency).
# Short acks played while the real reply generates. Several per language and
# rotated — a caller who hears the identical phrase every turn notices the seam
# immediately, which is worse than the silence it was meant to hide.
_DEFAULT_FILLER = {
    "el": ["Μάλιστα.", "Ένα λεπτό.", "Βεβαίως.", "Κατάλαβα."],
    "en": ["Okay, let me see.", "One moment.", "Sure, let me check.", "Right."],
    "ar": ["حسنًا.", "لحظة من فضلك.", "بالتأكيد."],
}

# Fixed opening line (no LLM) — pre-synthesized at startup for an instant greeting.
_DEFAULT_GREETING = {
    "el": "Γεια σας, καλέσατε την {company}. Πώς μπορώ να σας βοηθήσω;",
    "en": "Hello, you've reached {company}. How can I help you today?",
    "ar": "مرحبًا، لقد اتصلت بـ {company}. كيف يمكنني مساعدتك؟",
}

# Hidden instruction that seeds the opening line (persona §13 owns the wording).
_KICKOFF = {
    "el": "(Σύστημα: Μια εισερχόμενη κλήση μόλις συνδέθηκε. Χαιρέτησε τον καλούντα τώρα σύμφωνα με τις οδηγίες έναρξης, στα Ελληνικά.)",
    "en": "(System: An inbound call just connected. Greet the caller now per your opening guidelines.)",
    "ar": "(النظام: تم توصيل مكالمة واردة للتو. رحّب بالمتصل الآن وفقًا لإرشادات الافتتاح، بالعربية.)",
}


class EchoOrchestrator:
    def __init__(self, settings: Settings, asr: WhisperASR, tts: XttsClient) -> None:
        self._settings = settings
        self._asr = asr
        self._tts = tts

    def _primary(self) -> str:
        return self._settings.primary_language

    async def handle_call(self, session: CallSession) -> None:
        """Drive one call. Spawned per inbound connection by the SIP transport."""
        logger.info("Call %s connected", session.call_id)
        endpointer = SileroEndpointer(
            threshold=self._settings.vad_threshold,
            min_speech_ms=self._settings.vad_min_speech_ms,
            silence_ms=self._settings.vad_silence_ms,
        )
        # First turn auto-detects; afterwards we lock the language for the call.
        locked_lang: str | None = None

        try:
            await self._play(session, _GREETING[self._primary()], self._primary())

            prompted = False
            while not session.ended.is_set():
                utterance = await self._next_utterance(session, endpointer)
                if utterance is None:
                    # No-input timeout fired.
                    if prompted:
                        logger.info("Call %s: second timeout, ending", session.call_id)
                        break
                    prompted = True
                    await self._play(session, _STILL_THERE[locked_lang or self._primary()],
                                     locked_lang or self._primary())
                    continue
                prompted = False

                metrics = TurnMetrics(session.call_id)
                with metrics.stage("asr"):
                    result = await self._asr.transcribe(utterance, language=locked_lang)

                if result.is_empty:
                    metrics.log(skipped="empty_transcript", no_speech=result.no_speech_prob)
                    continue

                if locked_lang is None:
                    locked_lang = result.language
                    logger.info("Call %s: language locked to '%s'", session.call_id, locked_lang)

                reply = f"{_ECHO_PREFIX.get(locked_lang, _ECHO_PREFIX['en'])} {result.text}"
                with metrics.stage("tts"):
                    await self._play(session, reply, locked_lang)
                metrics.log(lang=locked_lang, chars=len(result.text))

        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - never let one bad call crash the service
            logger.exception("Call %s: unhandled error", session.call_id)
        finally:
            if not session.ended.is_set():
                await session.hangup()
            logger.info("Call %s: handler done", session.call_id)

    async def _next_utterance(
        self, session: CallSession, endpointer: SileroEndpointer
    ) -> np.ndarray | None:
        """Pull inbound audio, run VAD, return one complete utterance (or None on timeout)."""
        while True:
            try:
                chunk = await asyncio.wait_for(session.inbound.get(), timeout=_NO_INPUT_TIMEOUT_S)
            except asyncio.TimeoutError:
                return None
            if chunk is None:  # call ended
                return None
            event, audio = endpointer.process(chunk)
            if event == VadEvent.UTTERANCE_END and audio is not None:
                return audio

    async def _play(self, session: CallSession, text: str, language: str) -> None:
        """Synthesize and play, then wait for playback to drain."""
        try:
            pcm16 = await self._tts.synthesize_16k(text, language)
        except Exception:  # noqa: BLE001
            logger.exception("TTS failed for call %s", session.call_id)
            return
        await session.send_audio(pcm16)  # SIP bridge consumes 16 kHz PCM
        await self._drain(session)

    @staticmethod
    async def _drain(session: CallSession) -> None:
        # Wait until the outbound buffer empties, with a short tail of silence.
        while session.output_pending and not session.ended.is_set():
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.1)


class ConversationOrchestrator:
    """Phase-2 conversational loop: Gemma replies, streamed to TTS, with barge-in."""

    def __init__(
        self,
        settings: Settings,
        asr: WhisperASR,
        tts: XttsClient,
        llm: GemmaClient,
        system_prompt: str,
        mailer: Mailer | None = None,
        retriever: Retriever | None = None,
        system_prompts: dict[str, str] | None = None,
    ) -> None:
        self._settings = settings
        self._asr = asr
        self._tts = tts
        self._llm = llm
        self._system_prompt = system_prompt
        # Per-language rendered personas; falls back to the single prompt.
        self._system_prompts = system_prompts or {}
        self._mailer = mailer
        self._retriever = retriever
        self._greeting_pcm: np.ndarray | None = None  # cached at warmup()
        # Cached at warmup(), one per supported language — a call that locks to a
        # non-primary language must still get its ack, or the caller hears silence.
        self._filler_pcm: dict[str, list[np.ndarray]] = {}
        self._filler_idx = 0                           # rotates the ack variants

    def _primary(self) -> str:
        return self._settings.primary_language

    def _greeting_text(self, lang: str) -> str:
        if self._settings.greeting_text.strip():
            return self._settings.greeting_text.strip()
        template = _DEFAULT_GREETING.get(lang, _DEFAULT_GREETING["en"])
        return template.format(company=self._settings.company_name)

    async def warmup(self) -> None:
        """Pre-synthesize the greeting and warm the model stack (best-effort).

        Called once at startup so the first call's opening plays instantly and
        the first real turn doesn't pay cold-start latency on ASR/TTS.
        """
        try:
            pcm = await self._tts.synthesize_16k(
                self._greeting_text(self._primary()), self._primary()
            )
            lead_ms = self._settings.greeting_lead_silence_ms
            if lead_ms > 0:
                lead = np.zeros(16000 * lead_ms // 1000, dtype=np.int16)
                pcm = np.concatenate([lead, np.asarray(pcm, dtype=np.int16)])
            self._greeting_pcm = pcm
            logger.info("Greeting pre-synthesized (%d samples, +%dms lead)",
                        len(self._greeting_pcm), lead_ms)
        except Exception:  # noqa: BLE001 - fall back to per-call synth
            logger.warning("Greeting pre-synth failed; will synth per call", exc_info=True)
        if self._settings.thinking_filler_enabled:
            for flang in self._settings.language_list:
                variants = self._settings.thinking_filler_list or _DEFAULT_FILLER.get(flang, [])
                if not variants:
                    logger.warning("No thinking filler for '%s' — callers in that "
                                   "language hear silence while the reply generates", flang)
                    continue
                pcms: list[np.ndarray] = []
                for phrase in variants:
                    try:
                        pcms.append(await self._tts.synthesize_16k(phrase, flang))
                    except Exception:  # noqa: BLE001
                        logger.debug("Filler pre-synth failed for %s: %r",
                                     flang, phrase, exc_info=True)
                if pcms:
                    self._filler_pcm[flang] = pcms
                    logger.info("Filler pre-synthesized [%s]: %d variants %r",
                                flang, len(pcms), variants)
        try:  # warm faster-whisper (first transcription loads CUDA kernels)
            await self._asr.transcribe(np.zeros(16000, dtype=np.int16), language=self._primary())
        except Exception:  # noqa: BLE001
            logger.debug("ASR warmup skipped", exc_info=True)

    async def handle_call(self, session: CallSession) -> None:
        logger.info("Call %s connected (conversation)", session.call_id)
        endpointer = SileroEndpointer(
            threshold=self._settings.vad_threshold,
            min_speech_ms=self._settings.vad_min_speech_ms,
            silence_ms=self._settings.vad_silence_ms,
        )
        convo = Conversation(self._system_prompt, self._settings.llm_history_budget_tokens)
        lang = self._primary()        # locked to detected language after the first turn
        self._apply_language(convo, lang)  # pin language + persona; updated once ASR locks
        # Can we hand off to a human right now? Drives transfer vs. callback (§11).
        human_available = (
            self._settings.transfer_enabled
            and self._settings.has_transfer_target()
            and self._settings.within_business_hours()
        )
        # Use the caller's number from caller ID (if it looks like a real phone, not
        # an internal extension / withheld) so the callback flow needn't ask for it.
        caller_phone = session.caller_id if len(re.sub(r"\D", "", session.caller_id)) >= 9 else ""
        convo.set_availability(human_available, self._settings.transfer_departments(), caller_phone)
        convo.set_hours(self._settings.business_hours_text(), self._settings.within_business_hours())
        first_turn = True
        transferred = False
        emailed = False   # ensure exactly ONE ticket/email per call

        try:
            pending = await self._open(session, endpointer, convo, lang)

            prompted = False
            while not session.ended.is_set():
                if pending is not None:
                    utterance, pending = pending, None
                else:
                    utterance = await self._listen(session, endpointer)

                if utterance is None:  # no-input timeout
                    if prompted:
                        logger.info("Call %s: second timeout, ending", session.call_id)
                        break
                    prompted = True
                    pending = await self._speak(
                        session, endpointer, self._once(_STILL_THERE.get(lang, _STILL_THERE["en"])), lang
                    )
                    continue
                prompted = False

                metrics = TurnMetrics(session.call_id)
                with metrics.stage("asr"):
                    result = await self._asr.transcribe(
                        utterance, language=None if first_turn else lang
                    )
                if result.is_empty:
                    metrics.log(skipped="empty_transcript", no_speech=result.no_speech_prob)
                    continue
                if first_turn:
                    detected, prob = result.language, result.language_probability
                    if detected != self._primary() and prob < self._settings.asr_min_lang_prob:
                        # Low-confidence non-primary detection on a short clip →
                        # trust primary and re-transcribe so the text is right.
                        lang = self._primary()
                        logger.info(
                            "Call %s: detected '%s' (%.2f) below threshold → primary '%s'",
                            session.call_id, detected, prob, lang,
                        )
                        retry = await self._asr.transcribe(utterance, language=lang)
                        if not retry.is_empty:
                            result = retry
                    else:
                        lang = detected
                    first_turn = False
                    self._apply_language(convo, lang)
                    logger.info("Call %s: language locked to '%s' (agent '%s')",
                                session.call_id, lang,
                                self._settings.agent_name_for(lang))

                logger.info("Call %s HEARD [%s p=%.2f] %r",
                            session.call_id, lang, result.language_probability, result.text)
                convo.add_user(result.text)
                # Instant filler covers the ASR/RAG/LLM/TTS gap with a short ack.
                variants = self._filler_pcm.get(lang)
                if variants:
                    # Rotate rather than repeat — same phrase every turn is worse
                    # than none, because the caller learns to hear it as a stall.
                    await session.send_audio(variants[self._filler_idx % len(variants)])
                    self._filler_idx += 1
                context = ""
                if self._retriever is not None:
                    with metrics.stage("rag"):
                        context = await self._retriever.context_for(result.text)
                pending, reply, timing = await self._generate(
                    session, endpointer, convo, lang, context
                )
                spoken = _MARKER_RE.sub("", reply).strip()
                logger.info("Call %s REPLY [%s] %r", session.call_id, lang, reply)
                transfer_m = _TRANSFER_RE.search(reply)
                # Don't hang up on a turn that asks a question (e.g. "anything else?")
                # — the model often appends [[HANGUP]] anyway; wait for the answer.
                wants_hangup = _HANGUP_MARKER in reply and not spoken.endswith(("?", ";", "؟"))
                if _HANGUP_MARKER in reply and not wants_hangup:
                    logger.info("Call %s: ignoring [[HANGUP]] on a question turn", session.call_id)
                convo.add_assistant(spoken)  # store spoken text only
                department = (transfer_m.group(1) or "") if transfer_m else ""
                metrics.log(
                    lang=lang,
                    ttft_ms=timing.get("ttft"),
                    ttfa_ms=timing.get("ttfa"),
                    bargein=pending is not None,
                    transfer=department or bool(transfer_m),
                    hangup=wants_hangup,
                    chars=len(reply),
                )

                if transfer_m and human_available:
                    emailed = await self._drain_then_transfer(session, convo, lang, department)
                    transferred = True
                    # A blind transfer releases our leg (the call ends). If it's
                    # still up after the grace window, the PBX ignored/rejected the
                    # REFER — don't strand the caller in dead air: apologize, then
                    # fall through to hang up (the callback ticket is already sent).
                    try:
                        await asyncio.wait_for(
                            session.ended.wait(), timeout=self._settings.transfer_confirm_s
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Call %s: transfer not completed after %.0fs (PBX rejected REFER?) — falling back",
                            session.call_id, self._settings.transfer_confirm_s,
                        )
                        transferred = False
                        await self._speak(
                            session, endpointer,
                            self._once(_TRANSFER_FAILED.get(lang, _TRANSFER_FAILED["en"])), lang,
                        )
                    break
                if wants_hangup:
                    # Farewell already spoken+drained; end the call (finally hangs up).
                    logger.info("Call %s: agent ending call after farewell", session.call_id)
                    break

        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - one bad call must never crash the service
            logger.exception("Call %s: unhandled error", session.call_id)
        finally:
            # One ticket per call: skip the summary if the transfer already
            # emailed the same transcript (the escalation alert IS the ticket).
            if not emailed:
                await self._email_summary(
                    convo, lang, transferred, human_available, session.caller_id
                )
            if not session.ended.is_set() and not transferred:
                await session.hangup()
            logger.info("Call %s: handler done (transferred=%s)", session.call_id, transferred)

    # ── turn helpers ────────────────────────────────────────────────────────

    async def _open(
        self, session: CallSession, endpointer: SileroEndpointer, convo: Conversation, lang: str
    ) -> np.ndarray | None:
        """Speak the fixed opening line — instant if pre-synthesized at warmup."""
        text = self._greeting_text(lang)
        convo.add_assistant(text)
        if self._greeting_pcm is not None:
            # Instant: play cached audio, no LLM and no TTS round-trip.
            return await self._play_pcm(session, endpointer, self._greeting_pcm)
        # Fallback (warmup didn't cache / different lang): synth the fixed text now.
        pending, _, _ = await self._speak(session, endpointer, self._once(text), lang)
        return pending

    async def _play_pcm(
        self, session: CallSession, endpointer: SileroEndpointer, pcm: np.ndarray
    ) -> np.ndarray | None:
        """Play pre-rendered PCM while watching for barge-in (returns caller utterance)."""
        interrupt = asyncio.Event()
        captured: dict[str, np.ndarray | None] = {"audio": None}
        t0 = time.perf_counter()
        await session.send_audio(pcm)

        async def monitor() -> None:
            if not self._settings.bargein_enabled:
                return
            endpointer.reset()
            grace = self._settings.bargein_grace_ms / 1000.0
            while not session.ended.is_set():
                chunk = await session.inbound.get()
                if chunk is None:
                    return
                if time.perf_counter() - t0 < grace:
                    endpointer.reset()  # hold-off: don't flush the opening words
                    continue
                event, audio = endpointer.process(chunk)
                if event == VadEvent.SPEECH_START and not interrupt.is_set():
                    logger.info("Call %s: barge-in during greeting", session.call_id)
                    interrupt.set()
                    session.flush_output()
                if event == VadEvent.UTTERANCE_END:
                    captured["audio"] = audio
                    return

        mon = asyncio.create_task(monitor())
        try:
            while session.output_pending and not session.ended.is_set() and not interrupt.is_set():
                await asyncio.sleep(0.02)
            if not interrupt.is_set():
                await asyncio.sleep(0.1)
        finally:
            if interrupt.is_set():
                try:
                    await asyncio.wait_for(mon, timeout=_UTTERANCE_MAX_S)
                except asyncio.TimeoutError:
                    mon.cancel()
            else:
                mon.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await mon
        return captured["audio"]

    async def _drain_then_transfer(
        self, session: CallSession, convo: Conversation, lang: str, department: str = ""
    ) -> bool:
        """Wait for the handoff line to finish, alert the team, then SIP-REFER.

        Returns True if it emailed the call ticket (so the end-of-call summary is
        skipped — one ticket per call).
        """
        await self._drain(session)
        extension = self._settings.transfer_target(department)
        logger.info(
            "Call %s: transferring (department=%s → extension=%s)",
            session.call_id, department or "default", extension,
        )
        emailed = False
        if self._mailer and self._settings.email_escalation:
            dept_line = f" ({department})" if department else ""
            # Route the ticket to the department's email if one is configured.
            recipients = self._settings.escalation_email_for(department)
            emailed = await self._mailer.send(
                f"[Voice] Call transferred to {extension}{dept_line} — {self._settings.company_name}",
                self._summary_body(convo, lang, True, True, session.caller_id),
                recipients=recipients,
            )
        try:
            await session.transfer(extension)
        except Exception:  # noqa: BLE001 - transfer must not crash the call
            logger.exception("Call %s: transfer failed", session.call_id)
        return emailed

    @staticmethod
    async def _drain(session: CallSession) -> None:
        while session.output_pending and not session.ended.is_set():
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.1)

    async def _email_summary(
        self,
        convo: Conversation,
        lang: str,
        transferred: bool,
        human_available: bool,
        caller_id: str = "",
    ) -> None:
        if not (self._mailer and self._settings.email_summary):
            return
        if transferred:
            subject = f"[Voice] Call summary (transferred) — {self._settings.company_name}"
        elif not human_available:
            subject = f"[Voice] Callback needed (after-hours) — {self._settings.company_name}"
        else:
            subject = f"[Voice] Call summary — {self._settings.company_name}"
        await self._mailer.send(
            subject, self._summary_body(convo, lang, transferred, human_available, caller_id)
        )

    def _summary_body(
        self,
        convo: Conversation,
        lang: str,
        transferred: bool,
        human_available: bool,
        caller_id: str = "",
    ) -> str:
        header = [
            f"Company: {self._settings.company_name}",
            f"Client: {self._settings.client_id}",
            f"Caller ID: {caller_id or '— (withheld)'}",
            f"Language: {lang}",
            f"Transferred to human: {'yes' if transferred else 'no'}",
            f"Human available at call time: {'yes' if human_available else 'no'}",
        ]
        transcript = convo.transcript()
        if not transferred:  # callback case — surface captured contact details
            email, digits = _detect_contacts(transcript)
            # caller ID counts as a valid phone if it has enough digits
            cid_digits = re.sub(r"\D", "", caller_id)
            phone = digits if (digits and len(digits) >= 10) else (caller_id if len(cid_digits) >= 9 else "")
            header += [
                f"Auto-detected email: {email or '— (none/optional)'}",
                f"Phone for callback: {phone or '— MISSING — verify transcript'}",
            ]
        header += ["", "Transcript:"]
        return "\n".join(header) + "\n" + (transcript or "(no turns)")

    async def _generate(
        self,
        session: CallSession,
        endpointer: SileroEndpointer,
        convo: Conversation,
        lang: str,
        context: str = "",
    ) -> tuple[np.ndarray | None, str, dict]:
        """Stream a Gemma reply to TTS; on LLM failure, degrade gracefully."""
        messages = convo.messages_with_context(context) if context else convo.messages()
        try:
            return await self._speak(
                session, endpointer, self._llm.stream_chat(messages), lang
            )
        except LLMError:
            logger.exception("Call %s: LLM failed", session.call_id)
            return await self._speak(
                session, endpointer, self._once(_LLM_FAIL.get(lang, _LLM_FAIL["en"])), lang
            )

    async def _speak(
        self,
        session: CallSession,
        endpointer: SileroEndpointer,
        tokens: AsyncIterator[str],
        lang: str,
    ) -> tuple[np.ndarray | None, str, dict]:
        """Synthesize+play `tokens` clause-by-clause while watching for barge-in.

        Returns (barge_in_utterance | None, spoken_text, timing). If the caller
        interrupts, playback is flushed and their utterance is captured for the
        next turn. Re-raises LLMError from the token stream so callers can fall
        back. `timing` carries ttft/ttfa milliseconds for metrics.
        """
        interrupt = asyncio.Event()
        spoken: list[str] = []
        captured: dict[str, np.ndarray | None] = {"audio": None}
        timing: dict[str, float | None] = {"ttft": None, "ttfa": None}
        agg = ClauseAggregator(
            first_max_chars=self._settings.first_clause_max_chars
        )
        t0 = time.perf_counter()

        async def producer() -> None:
            try:
                async for delta in tokens:
                    if interrupt.is_set():
                        break
                    if timing["ttft"] is None:
                        timing["ttft"] = round((time.perf_counter() - t0) * 1000, 1)
                    spoken.append(delta)
                    for clause in agg.push(delta):
                        if interrupt.is_set():
                            break
                        await self._synth_and_send(session, clause, lang, interrupt, timing, t0)
                if not interrupt.is_set():
                    for tail in agg.flush_all():
                        if interrupt.is_set():
                            break
                        await self._synth_and_send(session, tail, lang, interrupt, timing, t0)
            finally:
                aclose = getattr(tokens, "aclose", None)
                if aclose is not None:
                    with contextlib.suppress(Exception):
                        await aclose()
            # Wait for playback to drain (interruptible).
            while session.output_pending and not session.ended.is_set() and not interrupt.is_set():
                await asyncio.sleep(0.02)
            if not interrupt.is_set():
                await asyncio.sleep(0.1)

        async def monitor() -> None:
            if not self._settings.bargein_enabled:
                return
            endpointer.reset()  # fresh VAD state for this listen-during-speak phase
            grace = self._settings.bargein_grace_ms / 1000.0
            while not session.ended.is_set():
                chunk = await session.inbound.get()
                if chunk is None:  # call ended
                    return
                if time.perf_counter() - t0 < grace:
                    endpointer.reset()  # hold-off: protect the reply's opening words
                    continue
                event, audio = endpointer.process(chunk)
                if event == VadEvent.SPEECH_START and not interrupt.is_set():
                    logger.info("Call %s: barge-in — stopping playback", session.call_id)
                    interrupt.set()
                    session.flush_output()
                if event == VadEvent.UTTERANCE_END:
                    captured["audio"] = audio
                    return

        prod = asyncio.create_task(producer())
        mon = asyncio.create_task(monitor())
        try:
            await prod
        finally:
            if interrupt.is_set():
                # Barge-in: let the monitor finish capturing the caller's utterance.
                try:
                    await asyncio.wait_for(mon, timeout=_UTTERANCE_MAX_S)
                except asyncio.TimeoutError:
                    mon.cancel()
            else:
                mon.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await mon

        return captured["audio"], "".join(spoken).strip(), timing

    def _apply_language(self, convo: Conversation, lang: str) -> None:
        """Pin the reply language AND swap in that language's persona, so the
        agent introduces itself by the right name (Rob in English, Nina in Greek)."""
        prompt = self._system_prompts.get(lang)
        if prompt:
            convo.set_base_system(prompt)
        convo.set_language(lang)

    async def _synth_and_send(
        self,
        session: CallSession,
        text: str,
        lang: str,
        interrupt: asyncio.Event,
        timing: dict,
        t0: float,
    ) -> None:
        text = _MARKER_RE.sub("", text).strip()  # never speak control tokens
        if not text or interrupt.is_set():
            return
        # Log exactly what goes to TTS. Without this, a bad value (e.g. a stray
        # .env comment leaking in as a filler) is invisible until you hear it.
        logger.info("Call %s SPEAK [%s] %r", session.call_id, lang, text)
        try:
            pcm16 = await self._tts.synthesize_16k(text, lang)
        except Exception:  # noqa: BLE001
            logger.exception("TTS failed for call %s", session.call_id)
            return
        if interrupt.is_set():  # barge-in landed during synthesis — drop it
            return
        if timing.get("ttfa") is None:
            # First audio of the turn. The RTP path swallows the opening moments
            # of a playback burst (the same reason the greeting carries a lead
            # pad), which clips the first phoneme. Pad only the first clause —
            # later clauses continue an already-open stream and need nothing.
            lead_ms = self._settings.reply_lead_silence_ms
            if lead_ms > 0:
                lead = np.zeros(16000 * lead_ms // 1000, dtype=np.int16)
                pcm16 = np.concatenate([lead, np.asarray(pcm16, dtype=np.int16)])
            timing["ttfa"] = round((time.perf_counter() - t0) * 1000, 1)
        await session.send_audio(pcm16)

    async def _listen(
        self, session: CallSession, endpointer: SileroEndpointer
    ) -> np.ndarray | None:
        """One complete inbound utterance, or None on no-input timeout / hangup."""
        endpointer.reset()  # start each turn's detection from clean VAD state
        logger.debug("Call %s: listening", session.call_id)
        timeout = self._settings.no_input_timeout_s
        while not session.ended.is_set():
            try:
                chunk = await asyncio.wait_for(session.inbound.get(), timeout=timeout)
            except asyncio.TimeoutError:
                return None
            if chunk is None:
                return None
            event, audio = endpointer.process(chunk)
            if event == VadEvent.UTTERANCE_END and audio is not None:
                return audio
        return None

    @staticmethod
    async def _once(text: str) -> AsyncIterator[str]:
        """Wrap a fixed string as a one-shot token stream (greeting/prompt/fallback)."""
        yield text
