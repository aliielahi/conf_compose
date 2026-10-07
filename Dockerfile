# GPU image for conf_compose. Match nvcc to the CUDA version of the PyTorch wheel:
# FlashInfer JIT compilation uses nvcc flags that CUDA 12.4 does not recognize.
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
        python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Dependencies live in the image; the project itself is mounted and installed at runtime.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch "transformers>=4.56" accelerate vllm \
        numpy scipy tqdm datasets math-verify python-dotenv pytest \
        matplotlib pandas

# Catch a future wheel/toolkit mismatch during the build, before model startup.
RUN python -c 'import re, subprocess, torch; nvcc = subprocess.check_output(["nvcc", "--version"], text=True); toolkit = re.search(r"release (\d+\.\d+)", nvcc).group(1); wheel = torch.version.cuda; print(f"nvcc CUDA {toolkit}; torch CUDA {wheel}"); assert toolkit == wheel, f"nvcc CUDA {toolkit} != torch CUDA {wheel}"'

WORKDIR /workspace
CMD ["sleep", "infinity"]
