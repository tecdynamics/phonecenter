# TecAI Voice Agent

Self-hosted, real-time **inbound telephone support agent**. Answers calls on a
Zadarma SIP extension, holds a multilingual conversation (Greek / English /
Arabic), and (later phases) answers from a per-client knowledge base via RAG.
One service instance per client.

Full spec: [`docs/voice_agent_build_spec.md`](docs/voice_agent_build_spec.md).
Voice persona prompt (loaded at runtime, not in code):
[`docs/voice_support_agent_prompt.md`](docs/voice_support_agent_prompt.md).

---

## Status — Phase 2 (full pipeline + barge-in, no RAG)

Phase 1 (spec §10.1) is done: register → answer → greeting → ASR → echo.
Phase 2 (spec §10.2) is now wired: **VAD + endpointing + Qwen/vLLM (streaming) +
XTTS + barge-in** on a live call, latency logged. The echo loop is kept as
`EchoOrchestrator` for diagnostics; `main.py` runs `ConversationOrchestrator`.

| Area                                                        | State |
|-------------------------------------------------------------|---|
| Config (`pydantic-settings`, per-client `.env`)             | ✅ |
| G.711 a-law/u-law codec (pure numpy, no `audioop`)          | ✅ unit-tested |
| 8k↔16k resampling (`soxr`)                                  | ✅ |
| Silero VAD endpointing                                      | ✅ (behind a swappable `Endpointer`) |
| faster-whisper ASR (large-v3, RTX 5070 / `auto` / CPU)      | ✅ verified loading on GPU |
| XTTS v2 TTS client                                          | ✅ — ⚠️ confirm the service HTTP contract |
| pjsua2 SIP transport (register/answer/bridge/DTMF/transfer) | ✅ code — ⚠️ **not yet built on tecai1** (`install_5070.sh` phase 3) |
| Qwen MoE LLM (vLLM `127.0.0.1:8000/v1`, streaming, history) | ✅ — ⚠️ validate on a live call |
| Streaming clause→TTS + per-turn LLM                         | ✅ unit-tested (clause splitter) |
| Barge-in (caller speech flushes TTS, captures utterance)    | ✅ code — ⚠️ tune AEC on a live call |
| RAG (e5 + vector store), DTMF/transfer flows                | ⏳ phases 3–4 |

### Things to confirm on the target box

1. **pjsua2 `MediaFrame` buffer marshalling** and the `AudioMediaPort` format
   API are version-dependent across PJSIP SWIG builds. See the caveats at the
   top of `voice_agent/sip/pjsip_endpoint.py` (`_frame_to_bytes` /
   `_bytes_into_frame`) — adjust to the installed pjsua2 if needed.
2. **XTTS v2 HTTP contract** (request fields, speaker selection, output
   format/rate). The assumption is isolated to `voice_agent/tts/xtts_client.py`
   — everything downstream only needs the int16 PCM it returns.
3. **LLM endpoint + chat template.** `LLM_ENDPOINT` points at the vLLM container
   on `127.0.0.1:8000` (OpenAI-compatible `/v1`), serving the Qwen MoE on the
   2× B70s. ⚠️ **Unlike llama.cpp, vLLM validates `model`** and returns 404 for
   an unknown name — `LLM_MODEL` must match the served id exactly:
   `curl -s http://127.0.0.1:8000/v1/models | jq -r '.data[].id'`.
   Also confirm the chat template accepts a system prompt followed by
   user/assistant turns.
4. **Barge-in vs. echo.** Barge-in runs VAD on inbound audio *during* playback
   and relies on pjmedia's acoustic echo canceller so the agent's own voice
   doesn't self-trigger. If the agent interrupts itself on a live call, raise
   `VAD_MIN_SPEECH_MS` / `VAD_THRESHOLD` or verify AEC is enabled.

---

## Architecture

**Everything runs on a single host, `tecai1`.** All model endpoints are loopback;
there is no remote model server.

- **RTX 5070, 12 GB** — the only CUDA device. Runs the voice agent process,
  faster-whisper ASR, XTTS, and e5 embeddings (phase 3), all in-process or over
  loopback.
- **2× B70** — runs the Qwen MoE under **vLLM in Docker**, reachable at
  `127.0.0.1:8000/v1`. These are not CUDA devices and do **not** appear in
  `nvidia-smi -L`, so `CUDA_VISIBLE_DEVICES=0` unambiguously means the 5070.

⚠️ **12 GB is a real budget.** Rough fp16 footprints: faster-whisper `large-v3`
≈3.1 GB, XTTS v2 ≈2.5–3 GB, multilingual-e5-large ≈1.3 GB (phase 3), plus
~0.3–0.5 GB CUDA context per process. That fits, but headroom is modest — watch
it when multi-call concurrency lands (phase 4), since XTTS buffers scale with
simultaneous synthesis. `large-v3-turbo` frees ~1.5 GB if needed, at some cost
to Greek/Arabic accuracy.

```
Zadarma PBX ──inbound──▶ [agent SIP extension]
                              │ pjsua2 UAC, 16 kHz bridge (pjmedia does G.711 + 8k↔16k)
                              ▼
   inbound 16k PCM ─▶ Silero VAD endpoint ─▶ faster-whisper ─▶ Qwen/vLLM (stream) ─▶ XTTS ─▶ 16k PCM out
                                  └─ during playback: caller speech ─▶ flush TTS (barge-in) ┘
```

The SIP layer sits behind `voice_agent/sip/base.py` (`SipTransport` /
`CallSession`) so a LiveKit SIP backend can replace pjsua2 without touching the
orchestrator.

```
voice_agent/
  config.py            pydantic settings, per-client load
  audio/               G.711 codec, soxr resampling, Silero VAD endpointer
  asr/                 faster-whisper client (RTX 5070)
  llm/                 Qwen/vLLM streaming client, clause splitter, conversation history
  tts/                 XTTS v2 HTTP wrapper
  sip/                 base.py (interface) + pjsip_endpoint.py (pjsua2 UAC)
  orchestrator.py      EchoOrchestrator (phase 1) + ConversationOrchestrator (phase 2, barge-in)
  metrics.py           per-stage latency logging
  main.py              service entrypoint
deploy/voice-agent.service    systemd unit (single instance; CUDA_VISIBLE_DEVICES=0 → the 5070)
tests/test_codec.py    G.711 codec checks (runnable anywhere)
```

---

## Install (target: `tecai1`)

**The scripted path is `./install_5070.sh`** — it does the venv, deps, a
faster-whisper GPU check, the PJSIP/pjsua2 build, and a starter `.env`. Prefer it
over the manual steps below, which are kept for reference and troubleshooting.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

⚠️ The venv is **Python 3.12**. Every `-devel` package below must match that
version, or the pjsua2 build fails looking for `Python.h`.

### Build pjsua2 (PJSIP Python bindings) — not on PyPI

`pjsua2` is not pip-installable; build it from PJSIP source. The bindings link
PJSIP's static libs, so the C side **must** be compiled with `-fPIC`. Keep the
venv active throughout so `make install` lands the module in `.venv`.

**1. Build dependencies.** `tecai1` is Debian/Ubuntu — use `apt`, as
`install_5070.sh` does:

```bash
apt-get update -y
apt-get install -y gcc g++ make pkg-config git swig python3.12-dev libssl-dev
```

`libssl-dev` is only needed for `SIP_TRANSPORT=tls`; `python3.12-dev` must match
the venv's Python exactly, or the build fails looking for `Python.h`.

**2. Fetch + build the PJSIP C libraries** (`--disable-sound`: we bridge audio
ourselves via `setNullDev` + a custom `AudioMediaPort`, so no ALSA is required):

```bash
cd /root
git clone --depth 1 --branch 2.15.1 https://github.com/pjsip/pjproject.git
cd pjproject
export CFLAGS="-fPIC -O2"
./configure --disable-sound
make dep && make
make install
ldconfig                                     # so the runtime linker finds the libs
```

**3. Build + install the Python module** (venv active → installs into `.venv`):

```bash
cd pjsip-apps/src/swig/python
make
make install
```

**4. Verify:**

```bash
python -c "import pjsua2; print('pjsua2 OK')"
```

Snags: `make` can't find `Python.h` → `python3.12-devel` missing or venv not
active (`which python` should be inside `.venv`). `import pjsua2` fails with a
`.so` load error → you skipped `ldconfig` or didn't set `CFLAGS="-fPIC"`.

faster-whisper (CTranslate2) and XTTS share the RTX 5070. The LLM does **not**
touch that GPU — it is a **Qwen MoE served by vLLM in Docker on the 2× B70s**,
OpenAI-compatible at `127.0.0.1:8000/v1`. The agent only *calls* it; it does not
start or manage the container.

> **Superseded:** earlier revisions of this document targeted a 6× Tesla V100
> AlmaLinux box, with the LLM on llama.cpp (`127.0.0.1:8084`), models split
> across GPUs 0–3 / 4 / 5, and a "vLLM is OUT" constraint justified by V100 being
> compute capability 7.0. None of that applies to `tecai1`, and vLLM **is** the
> LLM runtime now.

---

## Configure

```bash
cp .env.example .env     # or clients/<id>.env for systemd
# fill SIP_EXTENSION / SIP_PASSWORD and confirm model endpoints.
```

The agent never fills credentials for you. No secrets in code or git.

---

## Run

```bash
python -m voice_agent.main         # reads .env (or $ENV_FILE)
# production: systemctl enable --now voice-agent
```

## Validate Phase 2 (acceptance: a natural 3-turn chat with working interruptions)

1. Start the service; confirm `Registration state: code=200` in the logs.
2. Call the Zadarma DID/extension routed to the agent.
3. Hear the LLM-generated opening (persona §13; falls back to a fixed greeting
   if the LLM is unreachable).
4. Hold a short conversation — the agent answers via Qwen/vLLM, streamed to XTTS.
5. **Talk over the agent mid-sentence:** playback should stop immediately and
   the agent should respond to what you said (look for `barge-in` in the logs).
6. Check the per-turn metric line: `turn ... ttft_ms=… ttfa_ms=… bargein=…`.

> No RAG yet (phase 3): for factual questions the agent will correctly say it
> doesn't have the information and offer a human, per the persona grounding gate.

## Test (runnable without the model stack)

```bash
PYTHONPATH=. python tests/test_codec.py      # G.711 codec
PYTHONPATH=. python tests/test_clause.py     # LLM clause splitter
# or: pytest tests/
```

---

## Next

- **Phase 2 — done:** Qwen MoE (vLLM) streaming + **barge-in** + streaming TTS.
  Remaining: validate on a live call and tune barge-in/AEC.
- **Phase 3:** RAG (e5 + vector store) — *deferred to the end per current plan.*
- **Phases 4–5:** DTMF flows, SIP REFER transfer (wire the `_LLM_FAIL` path to an
  actual human handoff), multi-call hardening, second-client config.
