PYTHON ?= python
PIP ?= $(PYTHON) -m pip
IMAGE ?= matu-uq

.PHONY: install install-dev test syntax-check cli-help quick-eval paper-eval check build dist-check release-check clean docker-build docker-quick-eval

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

syntax-check:
	$(PYTHON) -m compileall matu baselines examples

cli-help:
	$(PYTHON) -m matu.cli --help

quick-eval:
	$(PYTHON) quick_start/code/04_evaluate_reference_results.py --sample math-qwen

paper-eval:
	$(PYTHON) quick_start/code/04_evaluate_reference_results.py --sample mmlu-autogen-qwen

check: test syntax-check cli-help quick-eval paper-eval

build:
	$(PYTHON) -m build

dist-check: build
	$(PYTHON) -m twine check dist/*

release-check: check dist-check

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -type f -delete
	find . -name '*:Zone.Identifier' -type f -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov build dist *.egg-info .coverage

docker-build:
	docker build -t $(IMAGE) .

docker-quick-eval:
	docker run --rm $(IMAGE) make quick-eval
