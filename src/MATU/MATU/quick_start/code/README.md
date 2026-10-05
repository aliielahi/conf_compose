# Quick Start Code

These scripts are small wrappers around the MATU package. The included result
files already let users evaluate the quick-start samples, so most scripts are
optional reproduction steps.

Run commands from the repository root.

| Script | What It Does | Required? | Output |
| --- | --- | --- | --- |
| `00_generate_logs_camel_gpt.py` | Example CAMEL/OpenAI log collector. | Optional; included logs are already provided. | New logs under `quick_start/generated/`. |
| `00_generate_logs_hf_qwen.py` | Example HF/Qwen log collector. | Optional; included logs are already provided. | New logs under `quick_start/generated/`. |
| `01_embed_reference_logs.py` | Stages the provided MATH embeddings by default, or recomputes Qwen3 embeddings from the included MATH conversation log when external model settings are enabled. | Optional; zipped reference embeddings are already included. | `quick_start/generated/embeddings/*.pkl`. |
| `02_run_cp2_from_generated_embeddings.py --embedding-source provided` | Runs CP-2/MATU on provided, generated, or explicitly passed role embeddings. | Optional; included fit curves are already in `quick_start/results/`. | `quick_start/generated/results/matu_scores.pkl` and `fit_dict_generated.pkl`. |
| `03_fit_to_uncertainty_reference.py` | Rebuilds the included MATH scalar uncertainty file from the included fit dictionary. | Optional; included uncertainty is already provided. | `quick_start/results/uncertainty_Math_qwen2.5.pkl`. |
| `03_fit_to_uncertainty_generated.py` | Converts generated fit curves into scalar MATU uncertainty. | Optional; only after generated CP-2. | `quick_start/generated/results/uncertainty_generated.pkl`. |
| `04_evaluate_reference_results.py --sample all` | Loads included MATU result files and labels, then prints the MATH and MMLU quick-start metrics. | Yes, for fastest verification. | Console AUROC/AUARC. |
| `04_evaluate_generated_results.py` | Evaluates regenerated uncertainty against included MATH labels. | Optional; verifies a regenerated pipeline run. | Console AUROC/AUARC. |
| `05_evaluate_baselines.py --sample all` | Loads the included SAUP-Multiple baseline score files and evaluates MATH plus MMLU AutoGen. | Optional. | Baseline AUROC/AUARC. |

To inspect or reuse the included embedding archives:

```bash
mkdir -p quick_start/generated/reference_embeddings/math
python -m zipfile -e quick_start/data/embeddings_Math_qwen2.5_qwen3.zip quick_start/generated/reference_embeddings/math

mkdir -p quick_start/generated/reference_embeddings/mmlu_autogen_qwen
python -m zipfile -e quick_start/data/embeddings_MMLU_Autogen_qwen2.5.zip quick_start/generated/reference_embeddings/mmlu_autogen_qwen
```

The included reference embeddings use `Qwen/Qwen3-Embedding-0.6B` and are
stored as zip archives instead of raw pickle files.

To rerun CP-2 with custom or freshly generated embeddings:

```bash
python quick_start/code/02_run_cp2_from_generated_embeddings.py \
  --embeddings path/to/user_embedding_matrices.pkl path/to/assistant_embedding_matrices.pkl
```
