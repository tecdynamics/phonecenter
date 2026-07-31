# Build Specification — AI Telephone Support Agent (TecAI)

**Audience:** an AI coding assistant building this project.
**Read this whole document before writing code. Do not rebuild infrastructure marked
"EXISTING — integrate, do not recreate." Ask before introducing any new external dependency
not listed here.**

---

## 1. GOAL

Build a self-hosted, real-time **inbound voice support agent** that answers phone calls routed
to a SIP extension on an existing Zadarma cloud PBX, holds a natural multilingual conversation
(Greek / English / Arabic), answers from a per-client knowledge base via RAG, and transfers to
a human when needed. One deployable service instance per client (ZAI, PetIQ, DIVIA, etc.),
configured by file/env, not by code changes.

Primary quality target: **end-to-end response latency (caller stops speaking → agent audio
starts) under 800 ms**, ideally ~500 ms. Everything else is secondary to keeping the call
feeling natural.

---

## 2. EXISTING INFRASTRUCTURE — integrate, do NOT recreate

These services already run on **`tecai1`**, which hosts the whole system. All endpoints are
**loopback** — there is no remote model server. Call them; do not reimplement them.

- **LLM:** a Qwen ~34B-parameter **MoE served by vLLM in Docker on 2× B70**, exposing an
  **OpenAI-compatible** `/v1/chat/completions` at `127.0.0.1:8000/v1`. Active and in
  production. Use it as the chat model.
- **TTS:** XTTS v2 on the RTX 5070 at `127.0.0.1:8087` (multilingual, voice cloning). Wrap its
  existing service; do not swap the model.
- **Embeddings:** `multilingual-e5-large` on the RTX 5070 at `127.0.0.1:8081`. Use it for RAG
  query embedding.
- **Telephony:** Zadarma **cloud PBX** with extensions and inbound routing already configured.
  The agent registers as **one new SIP extension**. **Do NOT install Asterisk/FreeSWITCH as a
  PBX** — Zadarma is the PBX. SIP credentials (server `sip.zadarma.com` or PBX host, extension
  number, password) are provided via config.

**New GPU work introduced by this project:** ASR — `faster-whisper large-v3` (CTranslate2) —
embedded in the agent process on the **RTX 5070**, alongside XTTS and (phase 3) e5.

**Hard platform constraints:**
- The **RTX 5070 (12 GB) is the only CUDA device.** The B70s running vLLM are not CUDA and
  never appear in `nvidia-smi -L`, so `CUDA_VISIBLE_DEVICES=0` unambiguously means the 5070
  and `ASR_DEVICE_INDEX=0` is the only valid index.
- **12 GB is a real budget.** Co-resident fp16: `large-v3` ≈3.1 GB, XTTS v2 ≈2.5–3 GB,
  e5-large ≈1.3 GB (phase 3), plus ~0.3–0.5 GB CUDA context per process. It fits with modest
  headroom — recheck at phase 4, where XTTS buffers scale with concurrent calls.
  `large-v3-turbo` frees ~1.5 GB at some cost to Greek/Arabic accuracy.
- **vLLM validates the model name** and returns 404 on a mismatch (llama.cpp, the previous
  backend, ignored it). `LLM_MODEL` must equal the id from `/v1/models` exactly.
- Telephony audio is **8 kHz a-law/u-law mono**. You must resample to/from 16 kHz PCM for
  ASR/TTS. Budget for the quality hit.

> **Superseded:** earlier revisions of this spec targeted a 6× Tesla V100 AlmaLinux box
> (ASRockRack ROMED8-2T, EPYC 7302) with Gemma on llama.cpp across GPUs 0–3, XTTS + e5 on
> GPU 4, ASR colocated on GPU 4, and Flux on GPU 5 — plus a "vLLM is out, V100 is compute
> capability 7.0" rule. None of that applies to `tecai1`, and vLLM **is** the LLM runtime now.

---

## 3. TARGET ARCHITECTURE

```
Zadarma PBX (existing, unchanged)
   └─ inbound rule routes call ──▶ [Agent extension 1XX]
                                        │  SIP REGISTER + RTP
                                        ▼
   ┌──────────────────── Agent service (this project, Python) ───────────────────┐
   │  SIP/RTP termination  ─▶  resample 8k→16k  ─▶  VAD + endpointing            │
   │        ▲                                              │                      │
   │        │ resample 16k→8k                              ▼                      │
   │   [XTTS v2]  ◀── TTS text ◀── [Qwen /v1/chat] ◀── prompt+RAG context        │
   │    (5070)                        (vLLM, B70)          ▲                      │
   │                                     │          [faster-whisper] (5070)       │
   │                              [e5 embeddings + vector store] ◀── transcript   │
   └──────────────────────────────────────────────────────────────────────────────┘
```

The orchestration layer owns the real-time state machine: VAD, end-of-utterance detection,
**barge-in** (stop TTS playback the instant the caller speaks), and the
listen→transcribe→retrieve→generate→speak loop.

---

## 4. TECH STACK

- **Language:** Python 3.11+, fully async (asyncio).
- **Orchestration framework:** **Pipecat** (frame-based pipeline, built-in VAD, turn-taking,
  and interruption handling). Implement each model as a Pipecat service/processor.
- **SIP/RTP termination (registers the Zadarma extension):** **pjsua2** (PJSIP Python
  bindings) — acts as a SIP UAC, REGISTERs the extension, answers inbound calls, and exposes
  PCM frames via a custom `AudioMediaPort`. Bridge those frames into the Pipecat pipeline.
  - *Alternative (note for the implementer, do not build both):* if telephony robustness
    (jitter, NAT, codec negotiation, DTMF, SIP REFER transfer) proves painful, the SIP layer
    can be replaced by a self-hosted **LiveKit SIP** service that registers the extension and
    bridges audio to the agent. Keep the SIP layer behind an interface so it's swappable.
- **VAD:** Silero VAD (via Pipecat) for speech detection and endpointing.
- **Audio:** `numpy` + a resampler (e.g. `soxr` / `librosa`) for 8k↔16k; handle a-law/u-law
  (G.711) decode/encode.
- **HTTP:** `httpx` (async) for the LLM, XTTS and e5 calls.
- **Config:** `pydantic-settings` + per-client `.env` / YAML.
- **Deploy:** packaged as a **systemd service** per client instance on `tecai1` (match the
  existing TecAI service pattern). Provide a unit file template. The venv is **Python 3.12**;
  pjsua2 must be built against that exact version (`./install_5070.sh`, phase 3).

Pin versions in `requirements.txt`. Do not add LangChain or heavyweight agent frameworks —
keep the LLM call a direct OpenAI-compatible request.

---

## 5. COMPONENT REQUIREMENTS

### 5.1 SIP / telephony termination
- REGISTER to the Zadarma PBX as the configured extension; auto re-register and reconnect on
  failure with backoff. Log registration state.
- Answer inbound calls; one active `CallSession` per call; support **multiple concurrent
  calls** (config: max concurrency).
- Decode inbound G.711 8 kHz → PCM16 16 kHz; encode outbound PCM16 → G.711 8 kHz.
- **DTMF:** capture RFC 2833 / inband digits and surface them to the orchestrator (used for
  menu fallback and verification flows).
- **Transfer:** support blind transfer to `{ESCALATION_EXTENSION}` (SIP REFER) for human
  handoff.
- Clean teardown on hangup from either side; release ASR/TTS resources.

### 5.2 ASR (faster-whisper)
- Load `large-v3` on the RTX 5070 (`ASR_DEVICE_INDEX=0`), int8_float16 or float16 (watch the
  12 GB budget); single shared model, per-call streams.
- Streaming/chunked transcription driven by VAD: emit interim results and a final transcript on
  end-of-utterance.
- Language: auto-detect on first utterance, then **lock** to that language for the call unless
  the caller clearly switches; expose detected language to the LLM and TTS layers. Restrict to
  {SUPPORTED_LANGUAGES}.
- Tune for short phone utterances; suppress hallucinated transcripts on silence.

### 5.3 Turn-taking, endpointing & barge-in
- VAD-based endpointing with configurable silence threshold (start ~500 ms, make it tunable).
- **Barge-in is mandatory:** when caller speech is detected during TTS playback, immediately
  stop playback, flush the outbound audio buffer, and start listening. Pipecat's interruption
  handling covers this — wire it up and verify on a real call.
- Handle the caller talking over a half-finished sentence without the agent restarting from
  scratch.

### 5.4 LLM (Qwen MoE via vLLM)
- Call the existing OpenAI-compatible endpoint at `127.0.0.1:8000/v1`.
  **temperature 0.35, top_p 0.9, max_tokens ≈220.**
- ⚠️ vLLM **validates** the `model` field — send the exact id from `/v1/models`, or every
  request 404s. Qwen "thinking" variants also stream reasoning as content, which would be
  spoken aloud on the call: send `chat_template_kwargs={"enable_thinking": false}`.
- System prompt is loaded from an **external file per client** (the voice-persona prompt — a
  separate deliverable; do not hardcode persona text here). Inject retrieved RAG context into
  the message sequence.
- Stream the completion token-by-token into the TTS layer to minimize time-to-first-audio —
  begin synthesizing as soon as the first sentence/clause is available.
- Maintain conversation history for the call; trim to a token budget.

### 5.5 RAG
- On each user turn that needs facts, embed the query with the e5 server (use e5's required
  `query:` / `passage:` prefixes), retrieve top-k (start k=4) from the per-client vector store.
- Vector store: per-client index; choose a lightweight self-hostable store (e.g. Qdrant or
  pgvector — pick one, justify, keep behind an interface). Provide an ingestion script that
  builds a client's index from their product/policy/order-FAQ source data.
- Assemble retrieved passages into the context block the system prompt expects. Pass **all
  relevant fields** through; do not silently drop data.

### 5.6 TTS (XTTS v2)
- Wrap the existing XTTS service as a Pipecat TTS processor. Stream audio out; target lowest
  possible time-to-first-audio.
- Per-client/per-language voice selection via config. Output resampled to 8 kHz for SIP.
- **Benchmark XTTS time-to-first-audio at 8 kHz early.** If it consistently adds >600 ms before
  speech, flag it and make the TTS layer swappable for a faster streaming TTS on the phone path.

### 5.7 Orchestration loop
- State machine: `idle → greeting → listening → transcribing → (retrieve) → generating →
  speaking → listening …`, with interruption transitions from `speaking → listening`.
- Silence handling: prompt once after no input; end politely after a second timeout.
- Optional answering-machine/voicemail detection for outbound (not required for inbound MVP).

---

## 6. LATENCY BUDGET (design to this)

| Stage | Target |
|---|---|
| VAD endpoint decision | ~300–500 ms (tunable) |
| ASR final transcript | <150 ms after endpoint |
| RAG retrieval | <100 ms |
| LLM time-to-first-token | <300 ms |
| TTS time-to-first-audio | <250 ms |

Overlap stages: start LLM streaming before ASR fully finalizes where safe; start TTS on the
first clause. Measure and log each stage per turn.

---

## 7. CONFIGURATION (per client)

Provide a single config object / `.env` covering at least:
`CLIENT_ID`, `COMPANY_NAME`, SIP (`SIP_SERVER`, `SIP_EXTENSION`, `SIP_PASSWORD`),
`SUPPORTED_LANGUAGES`, `PRIMARY_LANGUAGE`, `TTS_VOICE_<lang>`, `LLM_ENDPOINT`, `LLM_MODEL`,
`EMBEDDING_ENDPOINT`, `VECTOR_STORE_URL`, `RAG_INDEX`, `SYSTEM_PROMPT_PATH`,
`ESCALATION_EXTENSION`, `BUSINESS_HOURS`, `MAX_CONCURRENT_CALLS`, VAD/endpoint thresholds.

No secrets in code or git. The implementer must NOT enter credentials anywhere on the user's
behalf — provide the config template and let the user fill it.

---

## 8. SUGGESTED PROJECT STRUCTURE

```
voice_agent/
  config.py            # pydantic settings, per-client load
  sip/                 # pjsua2 UAC: register, call session, RTP<->PCM, DTMF, transfer
  audio/               # G.711 codec, resampling, buffers
  pipeline/            # Pipecat pipeline assembly + interruption wiring
  asr/                 # faster-whisper service (RTX 5070), streaming, lang detect
  llm/                 # Qwen/vLLM OpenAI-compatible client, streaming, history
  rag/                 # e5 embed client, vector store iface, retrieval, ingestion script
  tts/                 # XTTS wrapper, streaming, 8k output
  orchestrator.py      # call state machine, silence/escalation logic
  metrics.py           # per-stage latency + call logging
  main.py              # service entrypoint
deploy/voice-agent.service    # systemd unit (single instance; CUDA_VISIBLE_DEVICES=0 → the 5070)
requirements.txt
README.md
```

---

## 9. NON-FUNCTIONAL REQUIREMENTS

- **Concurrency:** handle N simultaneous calls (config), shared GPU models, per-call state
  isolation. No global mutable state across calls.
- **Resilience:** auto-reconnect SIP; graceful degradation if a model service is down (apologize
  + offer human transfer, never hang silently); never crash the service on a single bad call.
- **GDPR / privacy:** do not persist call audio by default; if recording is enabled, make it
  explicit config with a retention policy. Log transcripts only if configured, redact PII where
  feasible. The agent must not request or store full card numbers.
- **Observability:** structured logs per call (call id, language, turns, per-stage latency,
  escalations, errors). A simple health endpoint for the systemd service.
- **Security:** SIP over TLS / SRTP if Zadarma supports it for the account; restrict outbound;
  no credentials in logs.

---

## 10. DELIVER IN PHASES (acceptance criteria)

1. **SIP registration + echo:** agent registers the Zadarma extension, answers a call, and
   plays a fixed TTS greeting, then echoes back transcribed speech as TTS. Proves the audio
   loop end-to-end. ✅ when one clean turn works in Greek.
2. **Full pipeline, no RAG:** VAD + endpointing + barge-in + Qwen/vLLM + XTTS conversation on
   a live call, latency logged. ✅ when interruptions work and a 3-turn chat feels natural.
3. **RAG grounding:** wire e5 + vector store + ingestion; agent answers from a real client
   index and refuses/escalates when info is absent.
4. **Telephony features:** DTMF, human transfer (SIP REFER), silence handling, multi-call
   concurrency, systemd packaging.
5. **Hardening:** reconnection, error fallbacks, metrics, GDPR config, second-client config to
   prove the template generalizes.

Validate phase 1 before building phase 2. Do not build all phases blind.

---

## 11. KNOWN GOTCHAS (don't relearn these the hard way)

- No second PBX — register as an extension; Zadarma routes to it.
- 8 kHz narrowband in/out; all model audio is 16 kHz — resampling bugs cause garbled ASR/TTS.
- **The LLM is vLLM (Qwen MoE, port 8000).** The old "vLLM won't run on V100 (7.0), use
  llama.cpp" rule is void — it described the retired hardware and is now backwards.
- **vLLM validates `model`; llama.cpp did not.** A stale `LLM_MODEL` inherited from the
  llama.cpp era 404s every request. Read the id from `/v1/models`.
- **`CUDA_VISIBLE_DEVICES` pointing at a GPU that doesn't exist** yields *zero* visible
  devices, and CTranslate2 reports "sees no CUDA devices" rather than "bad index" — check it
  against `nvidia-smi -L` before suspecting drivers or libraries. Remember it also **remaps**:
  the selected GPU becomes ordinal 0 in-process, so `ASR_DEVICE_INDEX=0` is always right.
- **systemd `203/EXEC` means the file could not be executed at all** — wrong `ExecStart` path,
  missing venv, dangling interpreter symlink, CRLF line endings from a Windows edit, or no
  execute bit. It is never an application error; exit `1/FAILURE` is.
- **Edit `deploy/voice-agent.service` in the repo, not just `/etc/systemd/system/`** — the
  next `cp` silently reverts a fix applied only to the installed copy.
- **An unreachable XTTS does not crash startup.** The orchestrator logs "Greeting pre-synth
  failed; will synth per call" and continues, so the failure resurfaces once per live call.
- Barge-in and TTS time-to-first-audio are where "feels robotic vs natural" is won or lost.
- The persona/behavior system prompt is a SEPARATE file loaded at runtime — keep it out of code.
- Output sent to TTS must be plain spoken text (no markdown/symbols/URLs) — but that is the
  persona prompt's job, not the pipeline's; the pipeline just passes text through.
