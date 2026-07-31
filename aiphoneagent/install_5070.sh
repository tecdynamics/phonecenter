#!/usr/bin/env bash
# =====================================================================
# TecAI Voice Agent — install on the 5070 box (tecai1, Ubuntu 24.04)
# Run as root:   bash /tec/cuda/aiphoneagent/install_5070.sh 2>&1 | tee install.out
# Then paste install.out (it lands on the cuda share).
#
# Scope: phone-agent core = system deps -> Blackwell venv (cu128) ->
#        faster-whisper GPU check -> PJSIP/pjsua2 -> starter .env.
# TTS (:8087) and SIP creds are the follow-up step (see END).
# LLM is the LOCAL vLLM (127.0.0.1:8000, model 'tecai') — already running.
# =====================================================================
set -e
APP=/tec/cuda/aiphoneagent
PJ=/tec/cuda/pjproject
PY=python3.12
cd "$APP"

echo "############ PHASE 0: system deps (apt) ############"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y \
  ${PY}-venv ${PY}-dev \
  swig gcc g++ make pkg-config git \
  libspeexdsp-dev libssl-dev
echo "[phase0] swig: $(swig -version | grep -i version | head -1)"

echo "############ PHASE 1: venv + Blackwell PyTorch (cu128) + requirements ############"
# Blackwell (RTX 5070, sm_120) needs cu128 wheels. cu124 (the V100 doc) has NO
# sm_120 kernels and would fail at first CUDA op.
[ -d .venv ] || $PY -m venv .venv
# Use the venv by prepending it to PATH (works in sh/dash AND bash — no `source`).
export PATH="$APP/.venv/bin:$PATH"
python -V
pip install --upgrade pip wheel setuptools
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# If a dep (silero-vad) pulled a mismatched torch/torchaudio off PyPI, repair it.
# ⚠️ NOT --no-deps: torch's CUDA runtime ships as separate nvidia-*-cu12 wheels.
# --no-deps swaps torch to cu128 while leaving those at the version an earlier
# resolution installed (12.4), producing a cu128 torch on a 12.4 CUPTI that dies
# with "libtorch_cpu.so: undefined symbol: cuptiActivityEnableDriverApi".
# This is exactly how the chatterbox-server venv broke — do not reintroduce it.
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# speexdsp (AEC) — optional, no py3.12 build at the pinned version. Try 0.1.1,
# don't block: the AEC layer no-ops without it and barge-in stays off.
pip install "speexdsp==0.1.1" 2>/dev/null \
  && echo "[phase1] speexdsp OK (AEC available)" \
  || echo "[phase1] speexdsp unavailable on py3.12 -> AEC no-ops, barge-in stays off (fine for now)"
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("[phase1] torch", torch.__version__, "| cuda:", ok,
      "|", (torch.cuda.get_device_name(0) if ok else "NO GPU"))
print("[phase1] bf16 supported:", torch.cuda.is_bf16_supported() if ok else "n/a")
assert ok, "torch cannot see the GPU — stop here and check the cu128 install/driver"
PY

echo "############ PHASE 2: faster-whisper on the 5070 (CTranslate2 sm_120 risk) ############"
# This is THE known risk on brand-new Blackwell. If it fails with a CUDA kernel /
# 'no kernel image' error, the fix is ASR_DEVICE=cpu (works, a bit slower) or a
# newer/rebuilt CTranslate2. The install does NOT abort on this — it just reports.
set +e
python - <<'PY'
import numpy as np
try:
    from faster_whisper import WhisperModel
    # Dedicated 12GB -> quality over speed: full large-v3 (best Greek) at float16.
    m = WhisperModel("large-v3", device="cuda", device_index=0, compute_type="float16")
    segs, info = m.transcribe(np.zeros(16000, dtype=np.float32), language="el")
    list(segs)
    print("[phase2] WHISPER-GPU: OK (large-v3 float16 on cuda:0)")
except Exception as e:
    print("[phase2] WHISPER-GPU: FAILED ->", repr(e)[:300])
    print("[phase2] ACTION: set ASR_DEVICE=cpu in .env for now; report this line to me.")
PY
set -e

echo "############ PHASE 3: PJSIP 2.15.1 + pjsua2 (python3.12 SWIG bindings) ############"
cd /tec/cuda
[ -d "$PJ" ] || git clone --depth 1 --branch 2.15.1 https://github.com/pjsip/pjproject.git
cd "$PJ"
export CFLAGS="-fPIC -O2"
./configure --disable-sound          # we bridge audio ourselves; no ALSA
make dep && make && make install && ldconfig
cd pjsip-apps/src/swig/python
make
# NOTE: `make install` runs `setup.py install --user` -> installs to /root/.local,
# NOT the venv. Copy the freshly built module straight into the venv instead.
VSP="$APP/.venv/lib/python3.12/site-packages"
cp -f build/lib*/_pjsua2*.so "$VSP/"
cp -f build/lib*/pjsua2.py   "$VSP/"
# Verify from a neutral dir — importing from this build dir shadows the module.
cd /tmp && python -c "import pjsua2; print('[phase3] pjsua2 OK (in venv)')"
cd "$PJ/pjsip-apps/src/swig/python"

echo "############ PHASE 4: starter .env (LLM -> local vLLM; TTS/SIP to finish) ############"
cd "$APP"
if [ ! -f .env ]; then
  cp .env.example .env
  # point the LLM at the co-located vLLM (thinking OFF is handled in code)
  sed -i 's|^LLM_ENDPOINT=.*|LLM_ENDPOINT=http://127.0.0.1:8000/v1|' .env
  sed -i 's|^LLM_MODEL=.*|LLM_MODEL=tecai|' .env
  # ASR local on the 5070 — best Greek: full large-v3 at float16 (dedicated GPU)
  sed -i 's|^ASR_MODEL=.*|ASR_MODEL=large-v3|' .env
  sed -i 's|^ASR_DEVICE=.*|ASR_DEVICE=cuda|' .env
  sed -i 's|^ASR_DEVICE_INDEX=.*|ASR_DEVICE_INDEX=0|' .env
  sed -i 's|^ASR_COMPUTE_TYPE=.*|ASR_COMPUTE_TYPE=float16|' .env
  echo "[phase4] wrote .env (LLM->:8000/tecai, ASR->large-v3 float16 cuda:0)"
else
  echo "[phase4] .env already exists — left untouched"
fi

echo "############ DONE (core) ############"
echo "Next:"
echo "  1) faster-whisper result above — if it FAILED, tell me (CPU fallback or CT2 rebuild)."
echo "  2) TTS on :8087 is not installed yet — that's the next script."
echo "  3) Fill SIP_EXTENSION / SIP_PASSWORD in $APP/.env (I never fill secrets)."
echo "  4) Quick LLM smoke test from the venv:"
echo "     python - <<'PY'"
echo "     import asyncio; from voice_agent.llm.gemma_client import GemmaClient"
echo "     async def go():"
echo "       c=GemmaClient('http://127.0.0.1:8000/v1','tecai',chat_template_kwargs={'enable_thinking':False})"
echo "       async for t in c.stream_chat([{'role':'user','content':'Πες γεια σε μία πρόταση.'}]): print(t,end='')"
echo "       await c.aclose()"
echo "     asyncio.run(go())"
echo "     PY"
