#!/usr/bin/env bash
# Container for this project on the H200 server or the GB10 (DGX Spark), from NVIDIA's published vLLM image.
# Images: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm
set -euo pipefail

NAME=${CONTAINER_NAME:-uncertainty}
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: bash docker.sh COMMAND
  up                   Pull the image if needed; create/start the container; install datasets; check the GPU.
  shell                Enter the running container at /workspace.
  check                Test PyTorch GPU access, BF16 matmul and the vLLM import.
  install PACKAGE...   Install Python packages in the container, keeping NVIDIA's package constraints.
  stop                 Stop the container without deleting installed packages.
  status               Show the container and the image.
EOF
}

command_name=${1:-help}
shift || true
case "$command_name" in
  help|-h|--help) usage; exit 0 ;;
  up|shell|check|install|stop|status) ;;
  *) usage; exit 1 ;;
esac

[ "$(uname -s)" = Linux ] || fail "Run this on the GPU machine, not on your Mac."
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$gpu" in
  *GB10*)
    IMAGE=${IMAGE:-nvcr.io/nvidia/vllm:26.02-py3}
    MODEL_CACHE=${MODEL_CACHE:-$HOME/.cache/huggingface}
    ;;
  *H200*)
    IMAGE=${IMAGE:-nvcr.io/nvidia/vllm:26.03-py3}
    MODEL_CACHE=${MODEL_CACHE:-/mnt/data/hf-cache}
    mountpoint -q /mnt/data || fail "/mnt/data is not mounted."
    [ "$(docker info --format '{{.DockerRootDir}}')" = /mnt/data/docker ] ||
      fail "Docker storage must be /mnt/data/docker to keep it off the small root disk."
    ;;
  *) fail "Unknown GPU '$gpu'; set IMAGE and MODEL_CACHE by hand." ;;
esac

verify_container() {
  docker container inspect "$NAME" >/dev/null 2>&1 || fail "Container $NAME does not exist. Run: bash docker.sh up"
  local current_image workspace
  current_image=$(docker inspect --format '{{.Config.Image}}' "$NAME")
  workspace=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}' "$NAME")
  [ "$current_image" = "$IMAGE" ] || fail "$NAME uses $current_image, not $IMAGE."
  [ "$workspace" = "$PROJECT_DIR" ] || fail "$NAME belongs to another project: $workspace"
}

install() {
  docker exec "$NAME" bash -c '[ ! -f /etc/pip/constraint.txt ] || export PIP_CONSTRAINT=/etc/pip/constraint.txt
    python -m pip install --no-cache-dir -q "$@"' _ "$@"
}

check_gpu() {
  verify_container
  docker exec -i "$NAME" python - <<'PY'
import os
import torch
import vllm

print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda, "| vLLM:", vllm.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access the GPU")
print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
if not torch.all(x @ x.T == 64).item():
    raise RuntimeError("GPU matrix multiplication returned an incorrect result")
print("GPU BF16 matrix multiplication passed")
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
PY
}

case "$command_name" in
  up)
    mkdir -p "$MODEL_CACHE"
    [ -f "$PROJECT_DIR/.env" ] || echo "warning: no .env, so HF_TOKEN is unset; gated models (llama, gemma) will fail" >&2
    if docker container inspect "$NAME" >/dev/null 2>&1; then
      verify_container
      [ "$(docker inspect --format '{{.State.Running}}' "$NAME")" = true ] || docker start "$NAME"
    else
      docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
      docker run -d --name "$NAME" \
        --gpus all --ipc=host --init --restart unless-stopped \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --mount "type=bind,source=$PROJECT_DIR,target=/workspace" \
        --mount "type=bind,source=$MODEL_CACHE,target=/root/.cache/huggingface" \
        $([ -f "$PROJECT_DIR/.env" ] && echo "--env-file $PROJECT_DIR/.env") \
        --env HF_HOME=/root/.cache/huggingface \
        --env PYTHONPATH=/workspace/src \
        --env PYTHONDONTWRITEBYTECODE=1 \
        --env PYTHONUNBUFFERED=1 \
        --workdir /workspace \
        --entrypoint sleep "$IMAGE" infinity
      install datasets
    fi
    check_gpu
    printf '\nReady on %s. Enter with: bash docker.sh shell\n' "$gpu"
    ;;
  shell)
    verify_container
    docker exec -it "$NAME" bash
    ;;
  check)
    check_gpu
    ;;
  install)
    [ "$#" -gt 0 ] || fail "Specify packages, e.g. bash docker.sh install accelerate"
    verify_container
    install "$@"
    ;;
  stop)
    verify_container
    docker stop "$NAME"
    ;;
  status)
    docker ps -a --filter "name=^/${NAME}$"
    docker image ls "$IMAGE"
    ;;
esac
