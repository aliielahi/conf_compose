"""Project-wide constants loaded from constants.json, plus absolute project directories."""

import json
from pathlib import Path

CONSTANTS = json.loads((Path(__file__).parent / "constants.json").read_text())
SEED = CONSTANTS["seed"]
TASKS = CONSTANTS["tasks"]
SAMPLING = CONSTANTS["sampling"]
SEQUENCE_PROBABILITY = CONSTANTS["sequence_probability"]
EVALUATION = CONSTANTS["evaluation"]
VLLM = CONSTANTS["vllm"]
HF = CONSTANTS["hf"]
BASELINES = CONSTANTS["baselines"]

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / CONSTANTS["paths"]["results"]
LOGS_DIR = ROOT / CONSTANTS["paths"]["logs"]
CACHE_DIR = ROOT / CONSTANTS["paths"]["cache"]
