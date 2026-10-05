# GPU1 — machine notes

Everything about running `conf_compose` on this box. Written 2026-10-05.

## Hardware

| | |
|---|---|
| GPU | NVIDIA **H200 NVL**, 143 GB, single card |
| Driver | 595.91.07, reports CUDA 13.2 (runs any CUDA 12.x container) |
| Root disk | `/dev/sda1`, **39 GB** — small, fills easily |
| Data disk | `/mnt/data` (`h200-2-nvme1`), **14 TB** — put everything large here |

Roughly 4x the throughput of the old GB10 box. A candidate-scoring cell that took minutes there
takes ~30 seconds here.

## Where things live

```
/mnt/data/docker/                 Docker data-root: images, layers, build cache (moved off root)
/mnt/data/hf-cache/               Hugging Face model weights, ~60 GB for the seven models
/mnt/data/conf_compose/           the repo (move it here; the 39 GB root cannot hold results/)
    conf_compose/
        Dockerfile                the GPU image definition
        docker.sh                 build / up / shell / down
        .env                      HF_TOKEN etc. gitignored, never committed
        src/conf_compose/         all logic: tasks, backends, estimators, metrics, composition
        runs/inference/           the only code that loads a model
        runs/experiment*/         one experiment each, consuming the inference store
        results/inferences/       the inference store, grows to tens of GB
        results/<experiment>/     per-experiment outputs
        cache/                    LLM response cache, sqlite; makes reruns nearly free
        logs/                     per-run logs and DONE markers
        tests/                    regression tests
```

`results/`, `cache/`, `logs/` and `.env` are gitignored — they live only on disk, never in git.
Back them up with `rsync`, not `git`.

## One-time setup on a fresh machine

```bash
# 1. Docker onto the big disk (the 39 GB root cannot hold a vLLM image).
# /mnt/data is virtiofs on GPU1, so overlay2 cannot mount there; use fuse-overlayfs.
sudo apt-get update && sudo apt-get install -y fuse-overlayfs
sudo systemctl stop docker docker.socket      # the socket matters: it restarts the service
sudo mkdir -p /mnt/data/docker
sudo tee /etc/docker/daemon.json > /dev/null <<'EOF'
{
  "data-root": "/mnt/data/docker",
  "storage-driver": "fuse-overlayfs",
  "features": { "containerd-snapshotter": false }
}
EOF
sudo systemctl start docker
docker info | grep -E "Docker Root Dir|Storage Driver"
# must print /mnt/data/docker and fuse-overlayfs

# 2. model cache onto the big disk too
sudo mkdir -p /mnt/data/hf-cache
sudo chown "$USER":"$USER" /mnt/data/hf-cache
echo 'export HF_CACHE=/mnt/data/hf-cache' >> ~/.bashrc
export HF_CACHE=/mnt/data/hf-cache

# 3. the repo onto the big disk
mv ~/Desktop/conf_compose /mnt/data/conf_compose
cd /mnt/data/conf_compose/conf_compose

# 4. credentials
cp .env.example .env && nano .env             # HF_TOKEN=hf_...
```

`HF_TOKEN` is not optional: **gpqa, gemma and llama are gated** on the Hub. Without it, five of the
seven models and one dataset fail at load time.

## Daily use

```bash
cd /mnt/data/conf_compose/conf_compose
bash docker.sh build        # only after changing the Dockerfile; ~10-15 min
bash docker.sh up           # (re)create the container, install the package, verify
bash docker.sh shell        # same as: docker exec -it conf_compose bash
bash docker.sh down         # remove the container; results and cache are on the host, so nothing is lost
```

`up` prints a verification block. **All five lines must be right before running anything:**

```
nvcc ... release 13.0           <- matches torch CUDA below
gcc (Ubuntu 11.4.0) 11.4.0      <- C compiler present
torch 2.x cuda 13.0 gpus 1 | vllm 0.x
conf_compose ok
HF_TOKEN set: True
```

## Why the image is built the way it is

The base is `nvidia/cuda:13.0.0-devel-ubuntu22.04`. **devel, not runtime** — this is the single most
important detail. A runtime base has the driver and torch binaries but no toolchain, and then:

- `torch.compile` fails with *Failed to find C compiler* during kernel warmup
- FlashInfer fails with *Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist*

Both appear only once a model starts loading, minutes into a run, which makes them expensive to
discover. Shell workarounds (`export VLLM_USE_FLASHINFER_SAMPLER=0`) do not survive a new
`docker exec`, so the fix belongs in the image.

The CUDA compiler must also match the CUDA version bundled with PyTorch. On GPU1, the old
CUDA 12.4 image installed a CUDA 13.0 PyTorch wheel; FlashInfer then failed during model
warmup with `nvcc fatal: Unknown option '--compress-mode=size'`. The build now checks the
two versions and fails early if package updates make them diverge.

Container flags that matter, all set by `docker.sh up`:

| flag | why |
|---|---|
| `--gpus all` | obvious, but easy to forget |
| `--shm-size=32g` | Docker defaults to 64 MB; vLLM's EngineCore needs far more for IPC |
| `--ipc=host` | avoids shared-memory limits in multiprocess paths |
| `-v $PWD:/workspace` | code and results stay on the host and survive the container |
| `-v $HF_CACHE:/root/.cache/huggingface` | weights download once, not once per container |
| `--env-file .env` | HF_TOKEN reaches the process |

## Disk discipline

The root disk is 39 GB and the vLLM image alone is ~25 GB. If a build dies with
`[Errno 28] No space left on device`, that is why.

```bash
df -h /                     # root
df -h /mnt/data             # the big disk
docker system df            # what Docker is holding
docker builder prune -af    # build cache, usually the largest reclaimable chunk
docker system prune -af     # stopped containers and unused images
```

Keep `/mnt/data/hf-cache` — re-downloading seven models costs far more than the space it uses.

## Running work

Long jobs are detached and write a `DONE` marker, so disconnecting is safe:

```bash
bash runs/inference/sweep_full.sh gsm8k      # generation
cat current_run.log                          # stage, percentage, ETA; rewritten in full each update
cat logs/inference-gsm8k/DONE                # "ok" or "failed"
```

Useful env toggles on `sweep_full.sh`: `VOTERS`, `MODELS`, `TASKS`, `RETRY=4096`, `CANDIDATES=1`,
`SAMPLE_LOGPROBS=1`, `DRY_RUN=1`.

Auditors, all read-only and instant:

```bash
python runs/inference/check_store.py --task gsm8k --expect 35     # completeness, truncation, voter collisions
python runs/inference/check_lengths.py --tasks gsm8k --budget 2   # token limits from measured p98
python runs/inference/check_candidates.py --task csqa --target majority --match _full_
```

## Models

Seven in the store, all served through vLLM:

| alias | checkpoint | gated |
|---|---|---|
| `vllm/q3-4bi` | Qwen3-4B-Instruct-2507 | no |
| `vllm/q3-8bi` | Qwen3-8B | no |
| `vllm/l32-3bi` | Llama-3.2-3B-Instruct | **yes** |
| `vllm/l31-8bi` | Llama-3.1-8B-Instruct | **yes** |
| `vllm/g2-9i` | gemma-2-9b-it | **yes** |
| `vllm/g3-12i` | gemma-3-12b-it | **yes** |
| `vllm/phi4mii` | Phi-4-mini-instruct | no |

`q3-8bi` is a hybrid reasoning model: its chat template opens a `<think>` block, which breaks any
single-token probe. Handled by `CHAT_TEMPLATE_KWARGS` and `GENERATION_PREFIX` in
`src/conf_compose/utils/llm_calls/hf_models.py`. Any new reasoning model needs the same entry.

## Failure modes seen on this machine

| symptom | cause | fix |
|---|---|---|
| `No module named 'vllm'` | runtime deps missing | rebuild from the Dockerfile |
| `Failed to find C compiler` | runtime base, no gcc | devel base |
| `Could not find nvcc` | runtime base, no CUDA toolkit | devel base |
| `is a gated dataset` / 401 on a model | no HF_TOKEN | `.env` plus `--env-file` |
| `[Errno 28] No space left on device` | 39 GB root | Docker data-root on `/mnt/data` |
| `failed to mount overlay: invalid argument` | `/mnt/data` is virtiofs | use `fuse-overlayfs` and disable the containerd snapshotter |
| `nvcc fatal: Unknown option '--compress-mode=size'` | CUDA 12.4 compiler with CUDA 13.0 PyTorch/FlashInfer | rebuild from the CUDA 13.0 Dockerfile |
| `pull access denied for conf_compose` | image not built | `bash docker.sh build` |
| engine core init fails, no clear reason | 64 MB `/dev/shm` | `--shm-size=32g --ipc=host` |
