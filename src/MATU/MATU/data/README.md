# Data and Artifact Formats

MATU is designed to start from conversation logs that users have already
collected. The generation scripts in `examples/` are sample collectors, not a
required data source.

## Public Data Sources

The quick start includes copied conversation logs, labels, and reference
results, so users do not need to download benchmark data to verify the package.
The broader paper experiments use standard public benchmarks:

- MATH: Hendrycks et al. MATH benchmark. Official repository:
  https://github.com/hendrycks/math. The repository links to the Hugging Face
  mirror `qwedsacf/competition_math`, which can be loaded with
  `datasets.load_dataset("qwedsacf/competition_math")`.
- MMLU: Hugging Face dataset `cais/mmlu`:
  https://huggingface.co/datasets/cais/mmlu. Example:
  `datasets.load_dataset("cais/mmlu", "all")`, or replace `"all"` with a
  subject such as `"abstract_algebra"`.
- HumanEval / HumanEval+: EvalPlus repository:
  https://github.com/evalplus/evalplus. Install with `pip install evalplus`,
  then load with `from evalplus.data import get_human_eval_plus`.
- MoreHopQA: original Hugging Face dataset `alabnii/morehopqa`:
  https://huggingface.co/datasets/alabnii/morehopqa. The public dataset card
  recommends the human-verified split; a standard `datasets`-compatible mirror
  is available as `rdw79/morehopqa` and can be loaded with
  `datasets.load_dataset("rdw79/morehopqa", "verified")`.

Minimal download snippets:

```python
from datasets import load_dataset

math = load_dataset("qwedsacf/competition_math")
mmlu = load_dataset("cais/mmlu", "all")
morehopqa = load_dataset("rdw79/morehopqa", "verified")
```

```python
from evalplus.data import get_human_eval_plus

humaneval_plus = get_human_eval_plus()
```

The public framework is intentionally log-first: any agent framework can be
used as long as the resulting JSON follows `LOG_FORMAT.md`.

## Conversation Logs

See `LOG_FORMAT.md` for the required JSON structure. In short:

- Top-level keys are task or question ids.
- Each task id maps to repeated runs for the same input.
- Each run is an ordered list of turns.
- Each turn has a `role` and either an `output` or `content` string.

## Embedding Pickles

`matu embed` writes one pickle per role:

```text
<role>_embedding_matrices.pkl
```

Each file stores:

```python
dict[str, list[np.ndarray]]
```

The key is the task id. The list contains repeated runs. Each array has shape
`(num_steps_for_role, embedding_dim)`.

`embedding_metadata.json` records the embedding model, roles, and dimension.
The default public embedding model is `Qwen/Qwen3-Embedding-0.6B`.

## CP-2 / MATU Scores

`matu cp2` writes a pickle:

```python
dict[str, {
    "ranks": list[int],
    "fit": list[float],
    "loss": list[float],
    "relative_loss": list[float],
    "uncertainty": float,
}]
```

The scalar uncertainty is the sum of relative reconstruction losses across the
rank range. The optional `--legacy_fit_out` file stores the older paper-script
format:

```python
dict[str, list[float]]
```

where each list is the fit curve across ranks.

## Labels

Evaluation expects an `accuracy_dict` pickle:

```python
dict[str, list[str | int | bool]]
```

The values are repeated-run correctness labels. Accepted labels include
`correct`, `incorrect`, `1`, `0`, `true`, and `false`.

## Output Directories

CLI commands create their output directories automatically. The default config
uses `outputs/` for generated logs, embeddings, CP-2 results, scalar
uncertainty, and baseline scores. These generated outputs are ignored by git so
large local artifacts do not enter the public repository by accident.
