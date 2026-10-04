#!/usr/bin/env bash
# One-shot setup for a fresh GPU container: build toolchain, CUDA compiler, package, Hub auth.
# Usage: bash setup_container.sh          (idempotent; safe to re-run)
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n== %s\n' "$1"; }

say "system build toolchain (torch.compile needs a C compiler)"
if ! command -v gcc > /dev/null; then
  apt-get update -qq && apt-get install -y -qq build-essential
fi
gcc --version | head -1

say "python packages"
pip install -q -e ".[local]"

say "cuda compiler (FlashInfer JIT-builds its sampling kernel and needs nvcc)"
if ! command -v nvcc > /dev/null; then
  version=$(python -c "import torch; print((torch.version.cuda or '12').split('.')[0])")
  pip install -q "nvidia-cuda-nvcc-cu${version}" || echo "  no wheel for cu${version}; falling back below"
fi
nvcc_dir=$(python - <<'PY'
import os
try:
    import nvidia.cuda_nvcc as m
    print(os.path.dirname(m.__file__))
except Exception:
    print("")
PY
)
if [ -n "$nvcc_dir" ] && [ -d "$nvcc_dir/bin" ]; then
  echo "export CUDA_HOME=$nvcc_dir"        >> /etc/profile.d/conf_compose.sh
  echo "export PATH=$nvcc_dir/bin:\$PATH"  >> /etc/profile.d/conf_compose.sh
  export CUDA_HOME="$nvcc_dir" PATH="$nvcc_dir/bin:$PATH"
fi
if command -v nvcc > /dev/null; then
  nvcc --version | tail -1
else
  echo "  nvcc still missing: set VLLM_USE_FLASHINFER_SAMPLER=0 (slower sampling, works fine)"
  echo "export VLLM_USE_FLASHINFER_SAMPLER=0" >> /etc/profile.d/conf_compose.sh
fi

say "hugging face auth (gpqa, gemma and llama are gated)"
[ -f .env ] && set -a && . ./.env && set +a
if [ -z "${HF_TOKEN:-}" ]; then
  echo "  HF_TOKEN is not set. Put it in .env or pass -e HF_TOKEN=... to docker run."
  echo "  Without it: gpqa, g2-9i, g3-12i, l31-8bi and l32-3bi will all fail."
else
  python -c "from huggingface_hub import login; login('$HF_TOKEN', add_to_git_credential=False)" \
    && echo "  authenticated"
fi

say "checks"
python -c "import torch; print('  torch', torch.__version__, 'cuda', torch.version.cuda,
      'devices', torch.cuda.device_count())"
python -c "import vllm; print('  vllm', vllm.__version__)"
python -c "import conf_compose; print('  conf_compose importable')"
echo
echo "ready. Start the container with these so this survives a restart:"
echo "  docker run --gpus all --shm-size=32g --ipc=host \\"
echo "    -e HF_TOKEN=\$HF_TOKEN \\"
echo "    -v ~/.cache/huggingface:/root/.cache/huggingface \\"
echo "    -v \$PWD:/workspace/conf_compose <image>"
