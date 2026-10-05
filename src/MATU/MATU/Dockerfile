FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git make \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE CITATION.cff MANIFEST.in requirements.txt Makefile ./
COPY matu ./matu
COPY baselines ./baselines
COPY configs ./configs
COPY data ./data
COPY docs ./docs
COPY examples ./examples
COPY tests ./tests
COPY quick_start ./quick_start

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[dev]"

CMD ["matu", "--help"]
