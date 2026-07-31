#!/usr/bin/env bash
# Read-only assessment of the 5070 box. Changes nothing. Run and paste output.
set +e
echo "===== OS ====="
cat /etc/os-release 2>/dev/null | grep -E '^(NAME|VERSION)=' ; uname -r
echo "===== PACKAGE MANAGER ====="
command -v dnf && echo "-> dnf (RHEL/Alma)" ; command -v apt && echo "-> apt (Debian/Ubuntu)"
echo "===== PYTHON 3.11 ====="
for p in python3.11 python3.12 python3; do command -v $p >/dev/null && echo "$p -> $($p --version 2>&1)"; done
echo "===== BUILD TOOLS (for PJSIP) ====="
for t in gcc g++ make swig ldconfig pkg-config git; do printf "%-10s " "$t"; command -v $t >/dev/null && $t --version 2>&1 | head -1 || echo "MISSING"; done
echo "python dev headers:"; (python3.11 -c 'import sysconfig,os;print(os.path.join(sysconfig.get_path("include"),"Python.h"),"->", os.path.exists(os.path.join(sysconfig.get_path("include"),"Python.h")))' 2>/dev/null || echo "python3.11 not found")
echo "speexdsp dev (AEC):"; (ldconfig -p 2>/dev/null | grep -i speexdsp || echo "libspeexdsp NOT installed")
echo "===== CUDA / GPU ====="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
command -v nvcc >/dev/null && nvcc --version | tail -2 || echo "nvcc: no CUDA toolkit (ok - torch wheels bundle their own)"
echo "===== DISK (where we install) ====="
df -h /tec 2>/dev/null || df -h /opt 2>/dev/null; df -h "$HOME"
echo "===== LOCAL vLLM (the phone agent's LLM = 127.0.0.1:8000, model 'tecai') ====="
echo "-- /v1/models"; curl -sS -m 6 "http://127.0.0.1:8000/v1/models" 2>&1 | head -c 300; echo
echo "-- chat completion + does it LEAK reasoning? (bad for a phone: it'd be spoken)"
curl -sS -m 30 http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"tecai","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":60,"temperature":0.3}' \
  2>&1 | python3 -c "import sys,json
try:
    d=json.load(sys.stdin); m=d['choices'][0]['message']['content']
    print('REPLY:',repr(m)); print('USAGE:',d.get('usage'))
    print('THINKING-LEAK?', 'YES (see <think> or reasoning above)' if ('<think' in m.lower() or 'reasoning' in m.lower()) else 'not obviously')
except Exception as e: print('parse-fail:', e)"
echo "-- same with enable_thinking:false (this is what the agent will send)"
curl -sS -m 30 http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"tecai","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":60,"temperature":0.3,"chat_template_kwargs":{"enable_thinking":false}}' \
  2>&1 | python3 -c "import sys,json
try:
    d=json.load(sys.stdin); print('REPLY:',repr(d['choices'][0]['message']['content']))
except Exception as e: print('parse-fail (server may not accept chat_template_kwargs):', e)"
echo "===== what's already listening (8000/8084/8087/whisper) ====="
(ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) | grep -E ':(8000|8084|8087|8094|5060)\b' || echo "(none of 8000/8084/8087/8094/5060 listening)"
echo "===== DONE ====="
