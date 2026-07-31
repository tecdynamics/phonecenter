"""Per-client configuration via pydantic-settings.

Loaded from environment / a per-client .env file. No secrets are baked into
code or git — the operator fills .env (see .env.example).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Human-readable names for the {PRIMARY_LANGUAGE}/{SUPPORTED_LANGUAGES} placeholders.
_LANG_NAMES = {"el": "Greek", "en": "English", "ar": "Arabic"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # systemd template sets ENV_FILE=clients/<id>.env; default to .env locally.
        env_file=os.getenv("ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Identity ──────────────────────────────────────────────────────────
    client_id: str = "default"
    company_name: str = "TecAI"

    # ── SIP / Zadarma ─────────────────────────────────────────────────────
    sip_server: str
    sip_extension: str
    sip_password: str
    sip_transport: Literal["udp", "tcp", "tls"] = "udp"
    escalation_extension: str = ""   # default/fallback SIP REFER target
    # Department routing: "general:103,support:100,sales:302". The LLM picks the
    # department; we map it to an extension. Falls back to 'general' then
    # escalation_extension. Leave blank to always use escalation_extension.
    transfer_extensions: str = ""
    transfer_enabled: bool = True
    # After a transfer (SIP REFER), how long to wait for the call to actually drop
    # before treating the transfer as failed (PBX rejected/ignored REFER) and
    # falling back, so the caller is never stranded in silence.
    transfer_confirm_s: float = 15.0
    max_concurrent_calls: int = 2

    # ── Business hours (controls transfer vs. callback; empty = always open) ──
    business_open: str = ""          # "09:00"
    business_close: str = ""         # "17:00"
    business_days: str = "1-5"       # ISO weekdays Mon=1..Sun=7; "1-5" or "1,2,3"
    timezone: str = "Europe/London"

    # ── Email notifications (SMTP) ────────────────────────────────────────
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_to: str = ""                # comma-separated recipients (default ticket inbox)
    # Per-department ticket routing: "support:support@x.com,sales:sales@x.com,
    # general:ops@x.com". On a transfer, the ticket goes to the matching
    # department's email instead of SMTP_TO. Falls back to 'general' then SMTP_TO.
    escalation_emails: str = ""
    smtp_use_tls: bool = True
    smtp_verify_cert: bool = True    # set false for self-signed / hostname-mismatch certs
    email_summary: bool = True       # send a summary when each call ends
    email_escalation: bool = True    # alert the team when a call is transferred

    # ── Languages ─────────────────────────────────────────────────────────
    primary_language: str = "el"
    # CSV from env (e.g. "el,en"). Kept as a plain string so pydantic-settings
    # doesn't try to JSON-decode it — use `.language_list` for the parsed list.
    supported_languages: str = "el,en"

    # ── Model services ────────────────────────────────────────────────────
    # Qwen MoE via the EXISTING vLLM container on the 2x B70s (OpenAI-compatible)
    # at 127.0.0.1:8000. ⚠️ vLLM VALIDATES the model name and returns 404 on a
    # mismatch — llama.cpp, the previous backend, ignored it. Set llm_model from
    #     curl -s http://127.0.0.1:8000/v1/models | jq -r '.data[].id'
    llm_endpoint: str = "http://127.0.0.1:8000/v1"
    llm_model: str = "tecai"         # the id vLLM serves; must match /v1/models exactly
    llm_api_key: str = ""            # only if vLLM was started with --api-key
    llm_temperature: float = 0.35    # spec §5.4
    llm_top_p: float = 0.9
    llm_max_tokens: int = 110        # short replies = fewer/shorter TTS clauses = less lag
    llm_timeout: float = 30.0
    llm_history_budget_tokens: int = 4000  # trim history above this; raise only if ctx allows
    # Tighter length cap for the FIRST spoken clause only. TTS time scales with
    # clause length, and time-to-first-audio depends on that clause alone, so a
    # short opener gets the caller hearing a reply sooner; the rest synthesizes
    # while it plays. Too small sounds clipped — 60-90 is the useful band. 0 = off.
    first_clause_max_chars: int = 70
    # Qwen "thinking" models (the tecai vLLM) stream their reasoning as content
    # unless told not to. On a phone that reasoning gets SPOKEN, so keep this on;
    # it sends chat_template_kwargs={"enable_thinking": false}. llama.cpp ignores it.
    llm_disable_thinking: bool = True
    embedding_endpoint: str = "http://127.0.0.1:8081"
    embedding_model: str = "multilingual-e5-large"
    embedding_path: str = "/v1/embeddings"
    embedding_add_prefixes: bool = True   # e5 needs query:/passage: prefixes
    xtts_endpoint: str = "http://127.0.0.1:8087"   # TecAI tts-server (OpenAI-compatible)
    # Public, no-auth route (input capped at TTS_PUBLIC_MAX_CHARS server-side).
    # Switch to "/v1/audio/speech" + set xtts_api_key to use the token-gated route.
    xtts_path: str = "/v1/audio/speech/public"
    xtts_api_key: str = ""                          # Bearer token (only for the gated route)
    xtts_model: str = "tts-1"
    xtts_response_format: str = "wav"               # must be wav (parser depends on it)
    # Per-request TTS timeout. VoxCPM can take >15s on Greek (badcase retries),
    # which silently drops audio — raise so synthesis isn't lost. NB: a long
    # synth still means a long silence on the call; this is a safety net, not a
    # fix for slow TTS.
    xtts_timeout: float = 30.0
    # Speaking rate. <1.0 = slower/calmer, >1.0 = faster. Lower it if the voice
    # sounds rushed. (Timbre/childishness/noise come from the REFERENCE CLIP, not
    # this — fix those by re-cloning from a clean, calm adult recording.)
    xtts_speed: float = 1.0
    xtts_voice_el: str = "el_female_1"
    xtts_voice_en: str = "en_female_1"
    xtts_voice_ar: str = "ar_female_1"

    # ── ASR (faster-whisper) ──────────────────────────────────────────────
    asr_model: str = "large-v3"      # max el/en/ar quality, ~3.1 GB of the 5070's 12 GB
    asr_device: str = "cuda"
    # tecai1 has ONE CUDA device (RTX 5070) and the unit pins CUDA_VISIBLE_DEVICES=0,
    # so ordinal 0 is the only valid index. The B70s running vLLM are not CUDA.
    asr_device_index: int | Literal["auto"] = 0  # GPU index, or "auto" (freest)
    asr_compute_type: str = "int8_float16"
    # First-turn language lock: trust a NON-primary detection only at/above this
    # confidence; below it, fall back to primary (short clips mis-detect easily).
    asr_min_lang_prob: float = 0.85

    # ── Acoustic echo cancellation (AEC) ──────────────────────────────────
    # Cancels the agent's own TTS echoing back on the line so Whisper doesn't
    # transcribe phantom turns and barge-in doesn't flush the agent's reply.
    # Needs the `speexdsp` package on the server; falls back to a no-op (with a
    # warning) if it's missing. Prerequisite for re-enabling bargein_enabled.
    aec_enabled: bool = True
    # Adaptive filter length in ms — must exceed the worst-case round-trip echo
    # delay on the line (PSTN/cloud-PBX tails run ~100-250 ms). Longer = more CPU.
    aec_tail_ms: int = 200

    # ── VAD / endpointing ─────────────────────────────────────────────────
    # End-of-turn silence. Real VoIP/PSTN trunks have jitter + DTX gaps between
    # words; 450 ms chops callers mid-sentence on external calls. 800 is a safe
    # default — clean internal lines can go lower for snappier turns.
    vad_silence_ms: int = 800
    vad_threshold: float = 0.6    # higher = less likely to trigger on line echo/noise
    vad_min_speech_ms: int = 300  # ignore short noise blips on the trunk
    no_input_timeout_s: float = 6.0   # silence before "are you still there?" / ending
    # Short filler spoken instantly after the caller stops, while the real reply is
    # generated — masks TTS/LLM latency. Blank → a built-in default per language.
    thinking_filler_enabled: bool = True
    # Blank = built-in variants per language. Set a comma-separated list to
    # override for ALL languages, e.g. "One moment.,Let me check that."
    thinking_filler: str = ""

    @property
    def thinking_filler_list(self) -> list[str]:
        """Operator-supplied ack phrases; empty means use the built-in defaults."""
        return [x.strip() for x in self.thinking_filler.split(",") if x.strip()]
    # Barge-in: caller speech during playback stops TTS immediately (spec §5.3).
    # Still OFF by default: AEC (aec_enabled, above) now removes the line echo,
    # but its effectiveness must be VALIDATED on a live call before trusting
    # barge-in — without working echo cancellation the agent flushes its own
    # replies. Flip this on once a live test confirms AEC suppresses the echo.
    bargein_enabled: bool = False
    # Hold-off: ignore barge-in for the first N ms of each reply so the caller's
    # trailing audio / line echo can't flush the agent's opening words.
    bargein_grace_ms: int = 500

    # ── Greeting (fixed + pre-synthesized for an instant opening) ─────────
    # Spoken at call start with no LLM/TTS round-trip. Blank → a default that
    # uses {company} in the primary language. {company} is substituted.
    greeting_text: str = ""
    # Silence prepended to the greeting so the RTP/jitter-buffer ramp-up at call
    # connect doesn't clip the first word ("Hello" → "lo") or sound choppy.
    greeting_lead_silence_ms: int = 400
    # Same protection for every reply: the first clause of a turn gets a short
    # silence pad so the RTP ramp-up doesn't eat the opening consonant.
    reply_lead_silence_ms: int = 120

    # ── Persona prompt (separate runtime file) ────────────────────────────
    system_prompt_path: str = "docs/voice_support_agent_prompt.md"
    # Values substituted into the prompt's {PLACEHOLDER}s at load (per client).
    # company_name (above) fills {COMPANY_NAME}. Leave blank to keep a literal
    # {PLACEHOLDER} (and a startup warning) so you know it's unfilled.
    agent_name: str = ""              # {AGENT_NAME} — fallback for any language
    # Per-language personas. A Greek caller hears a Greek name, an English caller
    # an English one. Blank falls back to agent_name.
    agent_name_en: str = ""
    agent_name_el: str = ""
    business_domain: str = ""         # {BUSINESS_DOMAIN}
    business_hours: str = ""          # {BUSINESS_HOURS}
    supported_topics: str = ""        # {SUPPORTED_TOPICS}
    out_of_scope_topics: str = ""     # {OUT_OF_SCOPE_TOPICS}
    escalation_path: str = ""         # {ESCALATION_PATH} (defaults to the escalation extension)
    price_policy: str = "read_aloud"  # {PRICE_POLICY}
    followup_channel: str = "SMS"     # {FOLLOWUP_CHANNEL}
    verification_rule: str = ""       # {VERIFICATION_RULE}
    tools_available: str = ""         # {TOOLS_AVAILABLE}

    # ── RAG (phase 3) ─────────────────────────────────────────────────────
    rag_index: str = ""              # path to the .npz index; empty = RAG off
    rag_top_k: int = 4
    rag_min_score: float = 0.80      # e5 cosine threshold; below this = "not relevant"

    # ── Privacy ───────────────────────────────────────────────────────────
    record_audio: bool = False
    log_transcripts: bool = False

    @property
    def language_list(self) -> list[str]:
        """Parsed SUPPORTED_LANGUAGES, e.g. ['el', 'en', 'ar']."""
        return [x.strip() for x in self.supported_languages.split(",") if x.strip()]

    @property
    def smtp_recipients(self) -> list[str]:
        return [x.strip() for x in self.smtp_to.split(",") if x.strip()]

    def escalation_email_map(self) -> dict[str, str]:
        """Parse ESCALATION_EMAILS into {department: email}."""
        out: dict[str, str] = {}
        for part in self.escalation_emails.split(","):
            part = part.strip()
            if ":" in part:
                dept, email = part.split(":", 1)
                if dept.strip() and email.strip():
                    out[dept.strip().lower()] = email.strip()
        return out

    def escalation_email_for(self, department: str = "") -> list[str]:
        """Ticket recipients for a transfer: department email → 'general' → SMTP_TO."""
        m = self.escalation_email_map()
        if department and department.lower() in m:
            return [m[department.lower()]]
        if "general" in m:
            return [m["general"]]
        return self.smtp_recipients

    def business_hours_text(self) -> str:
        """Human-readable opening hours for the model to quote (empty if unset)."""
        if self.business_hours:        # operator's free-text wins
            return self.business_hours
        if self.business_open and self.business_close:
            return f"{self.business_days} {self.business_open}-{self.business_close} ({self.timezone})"
        return ""

    def transfer_map(self) -> dict[str, str]:
        """Parse TRANSFER_EXTENSIONS into {department: extension}."""
        out: dict[str, str] = {}
        for part in self.transfer_extensions.split(","):
            part = part.strip()
            if ":" in part:
                dept, ext = part.split(":", 1)
                if dept.strip() and ext.strip():
                    out[dept.strip().lower()] = ext.strip()
        return out

    def transfer_departments(self) -> list[str]:
        """Department names offered to the model (empty → single default target)."""
        depts = list(self.transfer_map().keys())
        if depts:
            return depts
        return ["general"] if self.escalation_extension else []

    def transfer_target(self, department: str = "") -> str:
        """Resolve a department to an extension, with sensible fallbacks."""
        m = self.transfer_map()
        if department and department.lower() in m:
            return m[department.lower()]
        if "general" in m:
            return m["general"]
        if self.escalation_extension:
            return self.escalation_extension
        return next(iter(m.values()), "")

    def has_transfer_target(self) -> bool:
        return bool(self.transfer_map() or self.escalation_extension)

    def _business_days(self) -> set[int]:
        days: set[int] = set()
        for part in self.business_days.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                days.update(range(int(a), int(b) + 1))
            elif part:
                days.add(int(part))
        return days or {1, 2, 3, 4, 5}

    def within_business_hours(self) -> bool:
        """True if a human is reachable now. Empty open/close → always open."""
        if not self.business_open or not self.business_close:
            return True
        from datetime import datetime
        from zoneinfo import ZoneInfo

        try:
            now = datetime.now(ZoneInfo(self.timezone))
        except Exception:  # noqa: BLE001 - bad tz name → use naive local time
            now = datetime.now()
        if now.isoweekday() not in self._business_days():
            return False

        def _minutes(hhmm: str) -> int:
            h, m = hhmm.split(":")
            return int(h) * 60 + int(m)

        cur = now.hour * 60 + now.minute
        return _minutes(self.business_open) <= cur < _minutes(self.business_close)

    def agent_name_for(self, lang: str | None = None) -> str:
        """Persona name for a language, falling back to the generic agent_name."""
        per_lang = {"en": self.agent_name_en, "el": self.agent_name_el}
        return per_lang.get((lang or "").lower(), "") or self.agent_name

    def persona_replacements(self, lang: str | None = None) -> dict[str, str]:
        """Map persona {PLACEHOLDER} names -> values from config/.env.

        Only non-empty values are returned; unset ones stay literal so the
        loader can warn the operator to fill them. `lang` selects the
        per-language persona name; None uses the generic one.
        """
        names = [_LANG_NAMES.get(x, x) for x in self.language_list]
        candidates = {
            "COMPANY_NAME": self.company_name,
            "AGENT_NAME": self.agent_name_for(lang),
            "BUSINESS_DOMAIN": self.business_domain,
            "BUSINESS_HOURS": self.business_hours,
            "SUPPORTED_TOPICS": self.supported_topics,
            "OUT_OF_SCOPE_TOPICS": self.out_of_scope_topics,
            "ESCALATION_PATH": self.escalation_path
            or (f"transfer to extension {self.escalation_extension}" if self.escalation_extension else ""),
            "PRICE_POLICY": self.price_policy,
            "FOLLOWUP_CHANNEL": self.followup_channel,
            "VERIFICATION_RULE": self.verification_rule,
            "TOOLS_AVAILABLE": self.tools_available,
            "PRIMARY_LANGUAGE": _LANG_NAMES.get(self.primary_language, self.primary_language),
            "SUPPORTED_LANGUAGES": ", ".join(names),
        }
        return {k: v for k, v in candidates.items() if v}

    def voice_for(self, lang: str) -> str:
        """Resolve the configured XTTS voice id for a language code."""
        return getattr(self, f"xtts_voice_{lang.lower()}", self.xtts_voice_en)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()  # type: ignore[call-arg]  # values come from env/.env
