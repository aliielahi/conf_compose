"""python -B -m voting_adapter.run --audit-only (from the CAGE-Cal repository)."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path

from .data import ROOT, DEFAULT_PROJECT, digest, file_hash, load_bundle, safe_output


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--panel", nargs="+", help="Ordered aliases, e.g. q3-4bi l31-8bi g3-12i; default all paper groups")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--audit-only", action="store_true", help="Validate sources and print counts; no files/models")
    mode.add_argument("--report-only", type=Path, metavar="RUN_DIR", help="Rebuild baseline-owned tables from a saved run")
    mode.add_argument("--finish-only", type=Path, metavar="RUN_DIR", help="Finish calibration and tables from saved seed predictions; no GPU")
    mode.add_argument("--smoke", action="store_true", help="Tiny synthetic graph training test; no downloads or real results")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--neighbors", type=int, default=20)
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--output-root", type=Path, default=ROOT / "results/voting_adapter")
    args = p.parse_args()
    if min(args.seeds, args.epochs, args.batch_size, args.neighbors) < 1 or not 0 < args.validation_fraction < 1:
        p.error("positive counts and 0 < --validation-fraction < 1 are required")
    return args


def contain_caches():
    cache = ROOT / ".cache-voting"
    for key, sub in (("HF_HOME", "huggingface"), ("HUGGINGFACE_HUB_CACHE", "huggingface/hub"),
                     ("HF_HUB_CACHE", "huggingface/hub"), ("HF_ASSETS_CACHE", "huggingface/assets"),
                     ("SENTENCE_TRANSFORMERS_HOME", "sentence_transformers"), ("TORCH_HOME", "torch"),
                     ("XDG_CACHE_HOME", "xdg"), ("CUDA_CACHE_PATH", "cuda"),
                     ("TRITON_CACHE_DIR", "triton"), ("MPLCONFIGDIR", "matplotlib"),
                     ("NUMBA_CACHE_DIR", "numba"), ("TMPDIR", "tmp")):
        path = safe_output(cache / sub)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    tempfile.tempdir = os.environ["TMPDIR"]
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def manifest_for(bundle, args):
    config = {key: getattr(args, key) for key in ("seeds", "epochs", "batch_size", "neighbors", "validation_fraction",
                                                "split_seed", "embedding_model")}
    config.update(tasks=sorted(bundle["splits"]), panels=sorted({g["models"] for g in bundle["groups"]}),
                  topology="iid", voter=0, response_logprob=True, brier_weight=0.4,
                  label_smoothing=0.05, min_calibration_questions=50)
    paths = sorted((ROOT / "voting_adapter").glob("*.py")) + sorted((ROOT / "cage_cal").glob("*.py"))
    paths += [ROOT / "scripts/cage_gnn_hypergraph.py", ROOT / "scripts/data_utils.py"]
    code = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    sources = dict(bundle["sources"])
    for rel in ("src/conf_compose/utils/metrics.py", "src/conf_compose/data/base.py", "src/conf_compose/data/numeric.py",
                "src/conf_compose/data/boolean.py", "src/conf_compose/data/multiple_choice.py",
                "paper_results/codes/voting_protocol/tables.py",
                "runs/experiment02-voting_composition/atomic.py",
                "src/conf_compose/composition/methods.py", "src/conf_compose/composition/candidates.py"):
        sources[rel] = file_hash(bundle["project"] / rel)
    return dict(schema=1, config=config, code=code, sources=sources, splits=bundle["splits"],
        groups=bundle["groups"], adaptations=[
            "IID: identical independent panel in both towers; communication adjacency is zero",
            "Voter 0 only; no extra rollouts or consistency samples used as graph nodes",
            "Task-aware equivalence and saved first/confidence/seeded voting rules match the paper manifest",
            "Graph ranks and plurality indicators mark the selected fixed target, including confidence-broken ties",
            "Internal validation is question-grouped within the existing outer fitting set",
            "W uses nearest training questions only, excludes self, one saved voter per model",
            "Answer PCA is training-only, zero-padded to 16 for small answer vocabularies",
            "Shared GNN trained jointly across selected tasks/panels, unlike per-panel pool fits",
            "Upstream graph architecture; preserve zero logprobs instead of truthiness fallback",
            "BetaSB uses internal validation; identity fallback for <50 unique questions or a degenerate fit",
            "Cons/seq rows share CAGE predictions but retain original matched evaluation masks"])


def runtime_info():
    packages = {}
    for name in ("torch", "torch-geometric", "sentence-transformers", "transformers", "numpy",
                 "scikit-learn", "betacal", "uncertainty-calibration"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, packages=packages)


def main():
    args = arguments()
    sys.dont_write_bytecode = True
    if args.smoke:
        contain_caches()
        from .test_adapter import graph_smoke
        graph_smoke(args.device)
        return
    previous = None
    if args.finish_only:
        saved_dir = safe_output(args.finish_only)
        previous = json.loads((saved_dir / "manifest.json").read_text())
        for key in ("seeds", "epochs", "batch_size", "neighbors", "validation_fraction", "split_seed", "embedding_model", "tasks"):
            setattr(args, key, previous["config"][key])
        panels = previous["config"]["panels"]
        args.panel = panels[0].split("|") if len(panels) == 1 else None
    bundle = load_bundle(args.project, args.tasks, args.panel, args.validation_fraction, args.split_seed)
    for g in bundle["groups"]:
        print(f"{g['task']} | {g['models']} | {g['counts']} | missing features={len(g['missing_feature_ids'])}")
    print(f"Audited {len(bundle['groups'])} panels and {len(bundle['cells'])} estimator-table cells; majority accuracy matches.")
    if args.audit_only:
        return
    safe_output(args.output_root)
    contain_caches()
    manifest = manifest_for(bundle, args)
    identity = digest(manifest)
    run_dir = safe_output(args.finish_only or args.report_only or (args.output_root / identity[:16]))
    if args.finish_only:
        # Code may change to repair postprocessing, but data/config cannot.
        for key in ("config", "sources", "splits", "groups"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Saved run {key} differs from current inputs; refusing recovery")
        from .training import finish_saved
        finish_saved(bundle, args, run_dir)
        (run_dir / "postprocessing.json").write_text(json.dumps(dict(
            training_identity=previous["identity"], code=manifest["code"], runtime=runtime_info(),
            mode="finish_saved_predictions_no_training"), indent=2))
        print("Recovered saved seed predictions; no training or model inference was run.")
    elif args.report_only:
        previous = json.loads((run_dir / "manifest.json").read_text())
        if previous["identity"] != identity:
            raise ValueError("Report request differs from saved manifest; use the original run arguments")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "manifest.json"
        if path.exists() and json.loads(path.read_text())["identity"] != identity:
            raise ValueError("Run directory contains a different experiment")
        path.write_text(json.dumps(dict(identity=identity, **manifest), indent=2))
        location = run_dir / "runtime.json"
        runtime = dict(project=str(bundle["project"]), device=args.device, **runtime_info())
        if location.exists() and json.loads(location.read_text())["packages"] != runtime["packages"]:
            raise ValueError("Dependency versions changed; choose a new --output-root instead of mixing checkpoints")
        location.write_text(json.dumps(runtime, indent=2))
        print(f"Run directory: {run_dir}", flush=True)
        from .training import run_training
        run_training(bundle, args, run_dir)
    from .report import make_reports
    rows = make_reports(bundle, run_dir / "predictions.jsonl", run_dir / "paper_tables")
    print(f"Wrote {len(rows)} CAGE metric rows and augmented TXT/LaTeX tables: {run_dir / 'paper_tables'}")


if __name__ == "__main__":
    main()
