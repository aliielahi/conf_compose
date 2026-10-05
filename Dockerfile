# GPU image for conf_compose. A -devel CUDA base ships nvcc and gcc, which vLLM's
# torch.compile and FlashInfer both JIT against; a -runtime base does not.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

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
        numpy tqdm datasets math-verify python-dotenv pytest \
        matplotlib pandas

WORKDIR /workspace
CMD ["sleep", "infinity"]
