# Chatterbox Multilingual TTS — production voice for the phone agent

**Chatterbox Multilingual** (resemble-ai, 0.5B LLaMA backbone, MIT) is the TTS
engine for the TecAI Voice Agent. It serves an OpenAI-compatible speech API on
**port 8087**, which is what `XTTS_ENDPOINT` in the agent's `.env` points at.

- Host: **`tecai1`**, path `/tec/cuda/chatterbox-server`
- GPU: the **RTX 5070 (12 GB)** — `CUDA_VISIBLE_DEVICES=0`, the only CUDA device
- Unit: `tecai-chatterbox.service` (log identifier `chatterbox-tts`)
- Greek (`language_id=el`) and English are both live

> **Naming note:** the agent's config keys and client module are still called
> `XTTS_*` / `xtts_client.py` for historical reasons. They talk to Chatterbox.
> The HTTP contract is what matters, not the name.

## Why Chatterbox (alternatives ruled out)
| Option | Greek | Cloning | Notes |
|--------|-------|---------|-------|
| XTTS v2 | ❌ not supported (17 langs, no `el`) | ✅ | would need a separate Greek engine again |
| Piper | ✅ | ❌ no cloning | loses the consistent custom voice |
| **Chatterbox Multilingual** | ✅ `el` | ✅ zero-shot, cross-lingual | 0.5B, ~5–7GB, MIT — covers all reqs in one engine |

## Platform rules — RTX 5070 (Blackwell, sm_120)
1. **torch must be a cu128 build.** Blackwell needs `sm_120` kernels; cu124
   wheels do not have them. Installed: `torch 2.11.0+cu128`. If a dependency
   drags torch off cu128, `torch.cuda.is_available()` goes False —
   `install_chatterbox_5070.sh` detects this and force-reinstalls.
2. **The venv is Python 3.12** (`.venv/`), matching the rest of the box.
3. `TORCHDYNAMO_DISABLE=1` stays set — torch.compile tracing has crashed on
   fresh torch builds.
4. `CHATTERBOX_FP16=false` for first-run audio correctness. Blackwell handles
   fp16/bf16 natively, so flipping it to `true` is a real latency win — re-check
   quality by ear before keeping it (vocoders can distort; the server only casts
   the backbone).
5. **VRAM is shared with ASR.** faster-whisper `large-v3` (~3.1 GB) runs in the
   agent process on the same card. Chatterbox is ~5–7 GB. That fits in 12 GB but
   leaves little slack — recheck when multi-call concurrency lands.

> **Superseded:** earlier revisions of this document described a pilot A/B against
> a production VoxCPM2 server on a 6× Tesla V100 box — port 8094, `venv/`,
> `/tec/ai/`, cu124 wheels, and "never bfloat16, Volta has no native bf16". None
> of that applies. Chatterbox won the evaluation and took over 8087; the V100
> host is retired and `ab_test.py` has been removed.

## Install
```bash
cd /tec/cuda/chatterbox-server
./install_chatterbox_5070.sh      # venv + cu128 torch + deps + GPU check
```

## Run it
```bash
sudo cp tecai-chatterbox.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tecai-chatterbox
journalctl -u tecai-chatterbox -f
```
First start downloads the 23-language checkpoint into `HF_HOME`
(`/tec/cuda/hf-cache`) — several GB, so expect a long pause before it serves.

Verify:
```bash
curl -s -o /tmp/tts.wav -w '%{http_code}\n' \
  http://127.0.0.1:8087/v1/audio/speech/public \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","input":"Καλημέρα","voice":"el_female_1","response_format":"wav","language":"el","speed":1.0}'
file /tmp/tts.wav      # expect: RIFF (little-endian) data, WAVE audio
```
`000` means nothing is listening; `curl` never reached the server.

## API
`POST /v1/audio/speech` (auth) and `/v1/audio/speech/public` (no auth) take the
OpenAI speech body plus two extensions — `language` (e.g. `el`, `en`) and TecAI
voice ids (`el_female_1`, `en_female_1`) instead of OpenAI's voice names. A
stock OpenAI TTS client will not work unmodified. Streaming variants are at
`/v1/audio/speech/stream[/public]`; `/health`, `/v1/models`,
`/v1/audio/languages` and `/v1/audio/voices` are also served.

Output is 8 kHz (`TTS_OUTPUT_SAMPLE_RATE=8000`) to match the phone path; the
agent resamples as needed.

## Tuning levers (speed/pacing)
- `CHATTERBOX_CFG_WEIGHT` (unit sets 0.6) — **lower = faster** pacing
- `CHATTERBOX_EXAGGERATION` (unit sets 0.4) — **higher = livelier + faster** rate
- `CHATTERBOX_TEMPERATURE` (unit sets 0.5) — lower = more consistent voice
- `TTS_SPEED` (unit sets 0.9) — overall speaking rate
- `CHATTERBOX_FP16` — try `true` for speed, verify quality by ear
- per-request overrides: `cfg_weight`, `exaggeration` in the JSON body

Voice consistency across calls comes from `TTS_LOCK_VOICE=en_female_1` plus
`TTS_VOICE_EL=el_female_1`, with reference wavs in `/tec/cuda/voices/presets/`.
