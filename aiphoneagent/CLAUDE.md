# TecAI Voice Agent

> ⚠️ The **global** `~/.claude/CLAUDE.md` describes unrelated projects (Zazo Clone,
> Siganos, Pagkritio Odeio, TecAI WebChat). **None apply here.** This file is the
> source of truth for this project: `T:\aiphoneagent`, which is a mount of
> `/tec/cuda/aiphoneagent` on the host **`tecai1`** — edits here are live on the server.

## What this is
Self-hosted, real-time **inbound telephone support agent** (Python 3.12, async).
Answers calls on a Zadarma SIP extension, holds a multilingual conversation
(Greek / English / Arabic), answers from a per-client knowledge base via RAG
(later phase), and transfers to a human when needed. **One service instance per
client** (ZAI, PetIQ, DIVIA, …), configured by `.env` — never by code changes.

Specs (read before non-trivial work):
- `docs/voice_agent_build_spec.md` — build/architecture spec
- `docs/voice_support_agent_prompt.md` — voice persona prompt (loaded at runtime, **not** hardcoded)

**Latency is the product:** target caller-stops-speaking → agent-audio-starts
**< 800 ms** (ideally ~500). Measure per stage (`metrics.py`).

## Build phases (spec §10) — validate each before the next
1. ✅ **SIP register + echo** (current): register → answer → Greek greeting → ASR → echo transcript as TTS. *Code complete; only the G.711 codec is unit-tested locally — the SIP/GPU path must be validated on the server.*
2. ⏳ Full pipeline: Qwen MoE (vLLM) + Pipecat + **barge-in** + streaming TTS.
3. ⏳ RAG (e5 + vector store) — **deferred to the end per user request.**
4. ⏳ DTMF, SIP REFER transfer, multi-call concurrency, systemd packaging.
5. ⏳ Hardening, metrics, GDPR config, second-client config.

## Architecture
```
Zadarma PBX ──inbound──▶ [agent SIP extension]
                              │ pjsua2 UAC, 16 kHz bridge (pjmedia does G.711 + 8k↔16k)
                              ▼
   inbound 16k PCM ─▶ Silero VAD endpoint ─▶ faster-whisper ─▶ (echo→LLM+RAG later) ─▶ XTTS ─▶ 16k PCM out
```
All in-pipeline audio is **mono int16 PCM at 16 kHz**. Telephony is 8 kHz G.711;
resampling lives at the edges. The SIP layer sits behind `sip/base.py`
(`SipTransport`/`CallSession`) so a LiveKit SIP backend can replace pjsua2.

## Structure
```
voice_agent/
  config.py        pydantic-settings, per-client .env (honors $ENV_FILE)
  audio/           codec.py (G.711, numpy), resample.py (soxr), vad.py (Silero endpointer)
  asr/             whisper_asr.py — faster-whisper, shared model, threaded
  tts/             xtts_client.py — async HTTP wrapper over existing XTTS
  sip/             base.py (interface) + pjsip_endpoint.py (pjsua2 UAC)
  orchestrator.py  phase-1 echo state machine
  metrics.py       per-stage latency
  main.py          entrypoint (one process per client)
deploy/voice-agent.service    systemd unit (single instance; CUDA_VISIBLE_DEVICES=0 → the 5070)
tests/test_codec.py           runs anywhere (no model stack needed)
```

## Existing infra — INTEGRATE, do NOT recreate
**Single host: `tecai1`.** Deployment path `/tec/cuda/aiphoneagent`. All model
endpoints are loopback — there is no remote model server.
- **LLM:** Qwen ~34B-param **MoE served by vLLM in Docker on 2× B70**, OpenAI-compatible at `127.0.0.1:8000/v1`. Active and in production. Sampling: temp 0.35, top_p 0.9, max_tokens ≈220. ⚠️ vLLM **validates** `LLM_MODEL` and 404s on a mismatch (llama.cpp did not) — take the id from `/v1/models`. Keep `llm_disable_thinking=True` or Qwen's reasoning gets spoken on the call.
- **TTS:** XTTS v2 on the 5070 at `127.0.0.1:8087` — wrap, don't swap. ⚠️ confirm its HTTP contract (`tts/xtts_client.py`).
- **Embeddings:** multilingual-e5-large on the 5070 at `127.0.0.1:8081` (use `query:`/`passage:` prefixes). (phase 3)
- **Telephony:** Zadarma **cloud PBX** is the PBX. Register as ONE extension. **No Asterisk/FreeSWITCH.**
- **ASR (new, this project):** faster-whisper, default **`large-v3`** (deliberate — max el/en/ar quality over turbo's speed), on the RTX 5070. `ASR_DEVICE_INDEX=0` is the only valid index. **Embedded in the agent process**, not a separate service.

### GPU budget — 12 GB total, and it matters
The RTX 5070 is the **only CUDA device**; the B70s are invisible to `nvidia-smi`
and never addressable via `CUDA_VISIBLE_DEVICES`. Co-resident fp16 footprints:
`large-v3` ≈3.1 GB, XTTS v2 ≈2.5–3 GB, e5-large ≈1.3 GB (phase 3), plus
~0.3–0.5 GB CUDA context per process. It fits, with modest headroom — revisit at
phase 4 (multi-call concurrency), where XTTS buffers scale with simultaneous synthesis.

> **Superseded — do not act on older docs.** Earlier revisions targeted a 6× Tesla
> V100 AlmaLinux box (LLM on llama.cpp GPUs 0–3, XTTS+e5 on GPU 4, ASR on GPU 5,
> ports 8084/8080). None of it applies. In particular the old "vLLM is OUT" rule
> was a V100 compute-capability-7.0 constraint and is now exactly backwards.

## Hard constraints / gotchas
- **SELF-CONTAINED ON `tecai1`. No TecAI main box, no V100, ever.** Need an LLM → the local B70 vLLM (`127.0.0.1:8000`). Need a GPU for anything else (ASR, TTS, embeddings) → the RTX 5070. Every model endpoint is loopback; a non-loopback model URL is a bug. The only legitimate outbound connections are `pbx.zadarma.com` (SIP registrar — that's the phone network) and `huggingface.co` for first-run checkpoint downloads (set `HF_HUB_OFFLINE=1` after caching to close even that).
- **vLLM is the LLM runtime** (Qwen MoE on the B70s, port 8000). The old "vLLM is OUT / stay on llama.cpp" rule applied to the retired V100 box and is void.
- `audioop` was **removed in Python 3.13** → G.711 is hand-rolled in `audio/codec.py` (numpy). G.711 clips at full scale (u-law max 32124) and a-law has no zero level (±8) — that's correct, not a bug.
- `large-v3-turbo` needs **faster-whisper ≥ 1.1.0**. Turbo is fast but weaker on non-English — **A/B Greek/Arabic** vs `large-v3`.
- **pjsua2 is not pip-installable** — build from PJSIP source with Python SWIG bindings on the server (README §Install). Two version-dependent spots flagged in `sip/pjsip_endpoint.py`: MediaFrame buffer marshalling + AudioMediaPort format API.
- **Barge-in** (stop TTS the instant the caller speaks) + TTS time-to-first-audio are where natural-vs-robotic is won (phase 2).
- **Persona prompt is a separate runtime file** (`SYSTEM_PROMPT_PATH`) — keep persona text out of code.
- **No secrets in code/git.** Never enter SIP/API credentials on the user's behalf — provide the template, they fill it.

## Commands
```bash
python -m voice_agent.main                  # run (reads .env or $ENV_FILE)
PYTHONPATH=. python tests/test_codec.py     # codec checks (or: pytest tests/)
python -m compileall voice_agent            # syntax check without the model stack
```
Dev box is Windows / Python 3.13; target is `tecai1`, Debian/Ubuntu / 3.12
(`T:\` is a mount of the server's `/tec/cuda`, so repo edits land live). The SIP + GPU path
cannot run on the dev box — validate Phase 1 on the server (README §Validate).
