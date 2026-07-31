#!/usr/bin/env bash
# =====================================================================
# Chatterbox Multilingual TTS — install on the 5070 (tecai1, Ubuntu 24.04)
# Local TTS for the phone agent, OpenAI-compatible, on port 8087.
# Run as root:  bash /tec/cuda/chatterbox-server/install_chatterbox_5070.sh 2>&1 | tee tts_install.out
# =====================================================================
set -e
APP=/tec/cuda/chatterbox-server
PY=python3.12
cd "$APP"

echo "############ deps: ffmpeg (audio post-processing) ############"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ffmpeg ${PY}-venv ${PY}-dev

echo "############ venv + Blackwell PyTorch (cu128) ############"
[ -d .venv ] || $PY -m venv .venv
# Use the venv via PATH (works in sh/dash AND bash — no `source`).
export PATH="$APP/.venv/bin:$PATH"
pip install --upgrade pip wheel setuptools
# Blackwell (sm_120): cu128. Install torch FIRST so chatterbox-tts sees it satisfied.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# chatterbox-tts may pull a non-cu128 torch as a dependency — verify & repair.
# ⚠️ Do NOT add --no-deps here. torch's CUDA runtime ships as separate
# nvidia-*-cu12 wheels; --no-deps swaps torch to cu128 but leaves those pinned at
# whatever the earlier resolution installed (12.4). The result imports as a cu128
# torch against a 12.4 CUPTI and dies with:
#   libtorch_cpu.so: undefined symbol: cuptiActivityEnableDriverApi
# Reinstalling WITH deps is what pulls the matching 12.8 nvidia wheels.
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "[fix] torch is off cu128 or its CUDA deps mismatch -> reinstalling WITH deps"
  pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
fi
# Fail loudly if the CUDA wheels are still on the 12.4 series under a cu128 torch.
python - <<'PY'
import sys
from importlib.metadata import version, PackageNotFoundError
try:
    cupti = version("nvidia-cuda-cupti-cu12")
except PackageNotFoundError:
    sys.exit(0)
if not cupti.startswith("12.8"):
    sys.exit(f"[fatal] nvidia-cuda-cupti-cu12 is {cupti}, expected 12.8.x — "
             "the CUDA wheels do not match the cu128 torch. Re-run the pip "
             "install above WITHOUT --no-deps.")
print("[cuda-wheels] cupti", cupti, "- matches cu128")
PY
python - <<'PY'
import torch
print("[torch]", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
assert torch.cuda.is_available(), "GPU not visible — stop and report"
PY

echo "############ dirs + HF cache ############"
mkdir -p /tec/cuda/media/audio /tec/cuda/hf-cache
export HF_HOME=/tec/cuda/hf-cache CHATTERBOX_DEVICE=cuda TORCHDYNAMO_DISABLE=1

echo "############ warmup + GREEK synth + latency (downloads model first run) ############"
python - <<'PY'
import time, torch, torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
t=time.time(); m=ChatterboxMultilingualTTS.from_pretrained(device="cuda")
print(f"[load] {time.time()-t:.1f}s  sr={m.sr}")
line="Καλησπέρα σας, καλωσορίσατε στην υποστήριξη. Πώς μπορώ να σας βοηθήσω;"
t=time.time()
wav=m.generate(line, language_id="el",
               audio_prompt_path="/tec/cuda/voices/presets/el_female_1.wav")
dt=time.time()-t
if wav.dim()==1: wav=wav.unsqueeze(0)
ta.save("/tec/cuda/chatterbox-server/greek_test.wav", wav.cpu().float(), m.sr, format="wav")
sec=wav.shape[-1]/m.sr
print(f"[greek] gen {dt:.2f}s for {sec:.2f}s audio  (RTF {dt/sec:.2f})  -> greek_test.wav")
print(f"[vram] peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
PY

echo "############ install + start service (port 8087) ############"
cp tecai-chatterbox.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now tecai-chatterbox
sleep 5
echo "-- health --"; curl -sS -m 15 http://127.0.0.1:8087/health || echo "(not up yet — check: journalctl -u chatterbox-tts -n 50)"
echo
echo "-- live Greek via the EXACT route the agent calls --"
curl -sS -m 60 http://127.0.0.1:8087/v1/audio/speech/public \
  -H "Content-Type: application/json" \
  -d '{"input":"Καλησπέρα, καλωσορίσατε στην υποστήριξη.","language":"el","voice":"el_female_1","response_format":"wav"}' \
  -o /tec/cuda/chatterbox-server/greek_route.wav && ls -la greek_route.wav || echo "route test failed"

echo "############ DONE ############"
echo "Listen on the share (T:\\chatterbox-server\\): greek_test.wav (direct) + greek_route.wav (via service)."
echo "Key numbers to report: [greek] gen time & RTF, [vram] peak, and health output."
echo "If Greek sounds good and RTF < ~0.5, the agent's TTS is ready (XTTS_ENDPOINT=http://127.0.0.1:8087)."
