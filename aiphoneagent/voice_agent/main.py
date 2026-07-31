"""Service entrypoint.

Wires config -> ASR + TTS + Gemma LLM -> conversation orchestrator -> pjsua2
SIP transport, then runs until SIGINT/SIGTERM. One process == one client
instance (systemd unit; see deploy/voice-agent.service).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from .asr.whisper_asr import WhisperASR
from .audio import SAMPLE_RATE_MODEL, SAMPLES_PER_FRAME_16K
from .audio.aec import EchoCanceller, build_echo_canceller
from .config import Settings, get_settings
from .llm.conversation import load_system_prompt
from .llm.gemma_client import GemmaClient
from .notify.mailer import Mailer
from .orchestrator import ConversationOrchestrator
from .rag.embeddings import EmbeddingClient
from .rag.retriever import Retriever
from .rag.store import LocalVectorStore
from .sip.base import CallSession
from .sip.pjsip_endpoint import Pjsua2Transport
from .tts.xtts_client import XttsClient

logger = logging.getLogger("voice_agent")


def _build_tts(settings: Settings) -> XttsClient:
    voices = {
        "el": settings.xtts_voice_el,
        "en": settings.xtts_voice_en,
        "ar": settings.xtts_voice_ar,
    }
    return XttsClient(
        endpoint=settings.xtts_endpoint,
        voices=voices,
        default_voice=voices.get(settings.primary_language, settings.xtts_voice_en),
        api_key=settings.xtts_api_key,
        model=settings.xtts_model,
        response_format=settings.xtts_response_format,
        path=settings.xtts_path,
        timeout=settings.xtts_timeout,
        speed=settings.xtts_speed,
    )


def _build_rag(settings: Settings) -> tuple[EmbeddingClient | None, Retriever | None]:
    """Build the retriever if a RAG index is configured and present."""
    if not settings.rag_index:
        logger.info("RAG disabled (RAG_INDEX not set).")
        return None, None
    if not os.path.exists(settings.rag_index):
        logger.warning("RAG_INDEX '%s' not found — running without RAG.", settings.rag_index)
        return None, None
    embedder = EmbeddingClient(
        endpoint=settings.embedding_endpoint,
        model=settings.embedding_model,
        path=settings.embedding_path,
        add_prefixes=settings.embedding_add_prefixes,
    )
    store = LocalVectorStore.load(settings.rag_index)
    retriever = Retriever(embedder, store, settings.rag_top_k, settings.rag_min_score)
    logger.info("RAG enabled from %s (top_k=%d, min_score=%.2f)",
                settings.rag_index, settings.rag_top_k, settings.rag_min_score)
    return embedder, retriever


async def run(settings: Settings) -> None:
    asr = WhisperASR(
        model=settings.asr_model,
        device=settings.asr_device,
        device_index=settings.asr_device_index,
        compute_type=settings.asr_compute_type,
        supported_languages=settings.language_list,
        primary_language=settings.primary_language,
    )
    tts = _build_tts(settings)
    llm = GemmaClient(
        endpoint=settings.llm_endpoint,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
        chat_template_kwargs=(
            {"enable_thinking": False} if settings.llm_disable_thinking else None
        ),
    )
    # One rendered persona per supported language so the agent introduces itself
    # with a language-appropriate name (e.g. Rob in English, Nina in Greek).
    system_prompts = {
        lang: load_system_prompt(
            settings.system_prompt_path, settings.persona_replacements(lang)
        )
        for lang in settings.language_list
    }
    system_prompt = system_prompts.get(settings.primary_language) or load_system_prompt(
        settings.system_prompt_path, settings.persona_replacements()
    )
    mailer = Mailer(
        host=settings.smtp_host,
        port=settings.smtp_port,
        username=settings.smtp_user,
        password=settings.smtp_password,
        sender=settings.smtp_from,
        recipients=settings.smtp_recipients,
        use_tls=settings.smtp_use_tls,
        verify_cert=settings.smtp_verify_cert,
    )
    embedder, retriever = _build_rag(settings)
    orchestrator = ConversationOrchestrator(
        settings, asr, tts, llm, system_prompt, mailer, retriever,
        system_prompts=system_prompts,
    )
    await orchestrator.warmup()  # pre-synthesize greeting + warm ASR/TTS for a fast first response

    tasks: set[asyncio.Task] = set()

    async def on_call(session: CallSession) -> None:
        task = asyncio.create_task(orchestrator.handle_call(session))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    def make_aec() -> EchoCanceller:
        # Fresh per-call canceller — the echo path differs per caller/line.
        return build_echo_canceller(
            enabled=settings.aec_enabled,
            frame_size=SAMPLES_PER_FRAME_16K,   # 320 = 20 ms @ 16 kHz (the bridge frame)
            tail_ms=settings.aec_tail_ms,
            sample_rate=SAMPLE_RATE_MODEL,
        )

    transport = Pjsua2Transport(
        on_call,
        sip_server=settings.sip_server,
        extension=settings.sip_extension,
        password=settings.sip_password,
        transport=settings.sip_transport,
        max_concurrent_calls=settings.max_concurrent_calls,
        aec_factory=make_aec,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows dev
            pass

    await transport.start()
    logger.info("Voice agent up for client '%s' (%s)", settings.client_id, settings.company_name)
    try:
        await stop.wait()
    finally:
        logger.info("Shutting down…")
        await transport.stop()
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await tts.aclose()
        await llm.aclose()
        if embedder is not None:
            await embedder.aclose()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    asyncio.run(run(settings))


if __name__ == "__main__":
    main()
