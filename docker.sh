#!/usr/bin/env bash
# Build the image, start the container with this folder as /workspace, or open a shell in it.
# Usage: bash docker.sh build | up | shell | down
set -euo pipefail
cd "$(dirname "$0")"

IMAGE=${IMAGE:-conf_compose:latest}
NAME=${NAME:-conf_compose}
# Model weights are tens of GB; keep them off the root disk.
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
# The large data disk, mounted at the same path so symlinks into it resolve on the host and in the container.
DATA_DIR=${DATA_DIR:-/mnt/data}

case "${1:-shell}" in
  build)
    docker build -t "$IMAGE" .
    ;;
  up)
    if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
      echo "image $IMAGE does not exist yet; run: bash docker.sh build" >&2
      exit 1
    fi
    [ -f .env ] || echo "warning: no .env, so HF_TOKEN is unset; gpqa, gemma and llama will fail" >&2
    docker rm -f "$NAME" 2> /dev/null || true
    mkdir -p "$HF_CACHE"
    docker run -d --name "$NAME" \
      --gpus all --shm-size=32g --ipc=host \
      -v "$PWD":/workspace \
      -v "$HF_CACHE":/root/.cache/huggingface \
      $([ -d "$DATA_DIR" ] && echo "-v $DATA_DIR:$DATA_DIR") \
      $([ -f .env ] && echo "--env-file .env") \
      -w /workspace "$IMAGE"
    docker exec "$NAME" pip install --no-cache-dir -q -e . --no-deps
    echo "started $NAME  (models cached in $HF_CACHE)"
    docker exec "$NAME" bash -lc 'nvcc --version | tail -1; gcc --version | head -1;
      python -c "import torch, vllm; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda,
        \"gpus\", torch.cuda.device_count(), \"| vllm\", vllm.__version__)";
      python -c "import conf_compose; print(\"conf_compose ok\")";
      python -c "import os; print(\"HF_TOKEN set:\", bool(os.environ.get(\"HF_TOKEN\")))"'
    ;;
  shell)
    docker exec -it "$NAME" bash
    ;;
  down)
    docker rm -f "$NAME"
    ;;
  *)
    echo "usage: bash docker.sh build | up | shell | down" >&2
    exit 1
    ;;
esac
