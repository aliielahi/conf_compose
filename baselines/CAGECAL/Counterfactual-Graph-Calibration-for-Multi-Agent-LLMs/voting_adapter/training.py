"""Reuse upstream graph construction/architecture with isolated data and caches."""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter

import numpy as np

from .data import ROOT, metric_module
from .features import answer_projection, query_matrices


def upstream():
    sys.path.insert(0, str(ROOT / "scripts"))
    import cage_gnn_hypergraph as graph
    return graph


def make_graph(row, matrix, answers):
    graph = upstream()
    agents = [model + "::voter" for model in row["model_ids"]]
    panel = {agent: {"answer": answer, "mean_logprob": lp}
             for agent, answer, lp in zip(agents, row["answers"], row["mean_logprobs"])}
    data = graph.panel_to_hyperdata(panel, panel, agents, matrix, matrix,
                                   np.zeros_like(matrix), answers, "iid")
    # Upstream uses `lp or -5.0`: retain a legitimate zero log-probability.
    import torch
    lp = torch.tensor((np.clip(row["mean_logprobs"], -10, 0) + 5) / 5, dtype=torch.float32)
    data.x_T[:, 0] = lp
    data.x_0[:, 0] = lp
    # Upstream Counter breaks ties by insertion order. Mark the actual fixed
    # paper target instead; otherwise the graph would describe another answer.
    counts = Counter(row["answers"])
    target = row.get("target", counts.most_common(1)[0][0])
    if target not in counts or counts[target] != max(counts.values()):
        raise ValueError("Fixed target is not a majority candidate")
    ordered = sorted(counts, key=lambda answer: (-counts[answer], answer != target))
    ranks = {answer: index for index, answer in enumerate(ordered)}
    for x in (data.x_T, data.x_0):
        x[:, 1] = torch.tensor([ranks[a] / max(len(counts) - 1, 1) for a in row["answers"]])
        x[:, 5] = torch.tensor([float(a == target) for a in row["answers"]])
    return data


def prepare_features(bundle, run_dir, args):
    from sentence_transformers import SentenceTransformer
    rows = bundle["rows"]
    questions = {(r["task"], r["id"]): r["question"] for r in rows}
    answer_texts = sorted({a for r in rows for a in r["answers"]})
    qkeys = sorted(questions)
    cache = run_dir / "embeddings.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as f:
            if f["answers"].tolist() != answer_texts or f["qkeys"].tolist() != [list(k) for k in qkeys]:
                raise ValueError("Embedding cache keys differ from manifest")
            aemb, qemb = f["answer_vectors"], f["question_vectors"]
    else:
        model = SentenceTransformer(args.embedding_model, device=args.device,
                                    cache_folder=str(ROOT / ".cache-voting/sentence_transformers"))
        aemb = model.encode(answer_texts, batch_size=128, normalize_embeddings=True,
                            show_progress_bar=True, convert_to_numpy=True)
        qemb = model.encode([questions[k] for k in qkeys], batch_size=64, normalize_embeddings=True,
                            show_progress_bar=True, convert_to_numpy=True)
        tmp = run_dir / "embeddings.partial.npz"
        np.savez_compressed(tmp, answers=np.asarray(answer_texts), qkeys=np.asarray(qkeys),
                            answer_vectors=aemb, question_vectors=qemb)
        tmp.replace(cache)
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    embeddings = dict(zip(answer_texts, aemb))
    projected, pca = answer_projection(embeddings, [a for r in rows if r["split"] == "train" for a in r["answers"]])
    np.savez_compressed(run_dir / "answer_pca.npz", **pca)
    matrices = query_matrices(rows, dict(zip(qkeys, qemb)), args.neighbors)
    datasets = {name: [] for name in ("train", "validation", "evaluation")}
    with (run_dir / "features.jsonl").open("w") as f:
        for row in rows:
            key = (row["task"], row["models"], row["id"])
            w, neighbors = matrices[key]
            data = make_graph(row, w, projected)
            datasets[row["split"]].append(dict(row=row, data=data))
            f.write(json.dumps(dict(task=row["task"], models=row["models"], id=row["id"],
                split=row["split"], target=row["target"], selection_reason=row.get("selection_reason"), answers=row["answers"],
                mean_logprobs=row["mean_logprobs"], training_neighbors=neighbors, W=w.tolist())) + "\n")
    return datasets


def collate(examples, tasks, device):
    import torch
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([e["data"] for e in examples]).to(device)
    bench = torch.zeros((len(examples), len(tasks)), device=device)
    for i, e in enumerate(examples):
        bench[i, tasks.index(e["row"]["task"])] = 1
    labels = torch.tensor([e["row"]["correct"] for e in examples], dtype=torch.float32, device=device)
    return batch, bench, labels


def predict(model, examples, tasks, device, batch_size):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch, bench, _ = collate(examples[start:start + batch_size], tasks, device)
            out.extend(torch.sigmoid(model(batch, bench)).cpu().tolist())
    return np.asarray(out)


def train_seed(datasets, tasks, args, seed, run_dir, metrics):
    import torch
    import torch.nn.functional as F
    graph = upstream()
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = graph.HyperHybridGNN(23, hid=64, heads=4, n_bench=len(tasks), dropout=0.3).to(args.device)
    checkpoint = run_dir / f"seed_{seed}.pt"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
        return model
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, best_state, patience = -float("inf"), None, 0
    train, val = datasets["train"], datasets["validation"]
    if not train or not val:
        raise ValueError("Both training and validation graph sets must be nonempty")
    yval = np.asarray([e["row"]["correct"] for e in val])
    benches = np.asarray([e["row"]["task"] for e in val])
    log = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        order = rng.permutation(len(train))
        for start in range(0, len(order), args.batch_size):
            chunk = [train[i] for i in order[start:start + args.batch_size]]
            batch, bench, y = collate(chunk, tasks, args.device)
            logits = model(batch, bench)
            loss = F.binary_cross_entropy_with_logits(logits, y * 0.9 + 0.05)
            loss = loss + 0.4 * F.mse_loss(torch.sigmoid(logits), y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        pval = predict(model, val, tasks, args.device, args.batch_size)
        scores = []
        for task in tasks:
            mask = benches == task
            if mask.any() and len(np.unique(yval[mask])) == 2:
                scores.append(metrics.auroc(pval[mask], yval[mask]) - 0.5 * metrics.ece(pval[mask], yval[mask]))
        # One-class validation fallback; evaluation labels are never used.
        score = float(np.mean(scores)) if scores else -metrics.brier(pval, yval)
        log.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), validation_score=score,
                        selection="auroc_minus_half_ece" if scores else "negative_brier"))
        print(f"seed={seed} epoch={epoch + 1} loss={np.mean(losses):.5f} val={score:.5f}", flush=True)
        if score > best:
            best, patience = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 5:
            break
    if best_state is None:
        raise ValueError("Training failed to produce a finite validation score")
    model.load_state_dict(best_state)
    tmp = run_dir / f"seed_{seed}.partial.pt"
    torch.save(best_state, tmp)
    tmp.replace(checkpoint)
    (run_dir / f"seed_{seed}_training.json").write_text(json.dumps(log, indent=2))
    return model


def calibrate(pval, peval, validation, evaluation, tasks, run_dir):
    """Upstream Beta+Platt-binning ensemble, fitted on internal validation only.

    Require 50 distinct questions, not 50 overlapping panel rows, per task.
    Missing/degenerate calibrators fall back explicitly to the raw prediction.
    """
    from betacal import BetaCalibration
    from calibration import PlattBinnerCalibrator
    output = peval.copy()
    states = {}
    for task in tasks:
        vi = np.asarray([e["row"]["task"] == task for e in validation])
        ti = np.asarray([e["row"]["task"] == task for e in evaluation])
        qids = {e["row"]["id"] for e in validation if e["row"]["task"] == task}
        yy = np.asarray([e["row"]["correct"] for e in validation])[vi]
        states[task] = dict(n_questions=len(qids), n_rows=int(vi.sum()), components=[], errors=[])
        if len(qids) < 50 or len(np.unique(yy)) < 2:
            states[task]["fallback"] = "identity: fewer than 50 validation questions or one class"
            continue
        pp = np.clip(pval[vi], 1e-6, 1 - 1e-6)
        predictions = []
        for name in ("beta", "platt_binner"):
            try:
                if name == "beta":
                    cal = BetaCalibration(parameters="abm").fit(pp.reshape(-1, 1), yy)
                    pred = cal.predict(np.clip(peval[ti], 1e-6, 1 - 1e-6).reshape(-1, 1)).ravel()
                else:
                    cal = PlattBinnerCalibrator(num_calibration=len(yy), num_bins=10)
                    cal.train_calibration(pp, yy.astype(int))
                    pred = np.asarray(cal.calibrate(peval[ti]), dtype=float)
                if not np.isfinite(pred).all() or ((pred < 0) | (pred > 1)).any():
                    raise ValueError("Calibration returned invalid probabilities")
                predictions.append(pred)
                states[task]["components"].append(name)
            except (ValueError, RuntimeError, FloatingPointError, AssertionError, np.linalg.LinAlgError) as exc:
                states[task]["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
        if predictions:
            output[ti] = np.mean(predictions, axis=0)
        else:
            states[task]["fallback"] = "identity: both calibration fits failed"
    (run_dir / "calibration.json").write_text(json.dumps(states, indent=2))
    # PlattBinnerCalibrator contains a local closure and cannot be pickled.
    # Persist numerical replay inputs instead of interpreter-specific objects.
    tmp = run_dir / "calibration_inputs.partial.npz"
    np.savez_compressed(tmp, validation_probabilities=pval, evaluation_probabilities=peval,
        validation_labels=np.asarray([e["row"]["correct"] for e in validation]),
        validation_tasks=np.asarray([e["row"]["task"] for e in validation]),
        validation_ids=np.asarray([e["row"]["id"] for e in validation]),
        evaluation_tasks=np.asarray([e["row"]["task"] for e in evaluation]),
        evaluation_ids=np.asarray([e["row"]["id"] for e in evaluation]),
        calibrated_probabilities=output)
    tmp.replace(run_dir / "calibration_inputs.npz")
    return output


def run_training(bundle, args, run_dir):
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run --audit-only locally and train on the GPU")
    metrics = metric_module(bundle["project"])
    started = time.monotonic()
    datasets = prepare_features(bundle, run_dir, args)
    prepared = time.monotonic()
    tasks = sorted(bundle["splits"])
    val_preds, eval_preds = [], []
    for seed in range(args.seeds):
        model = train_seed(datasets, tasks, args, seed, run_dir, metrics)
        pv = predict(model, datasets["validation"], tasks, args.device, args.batch_size)
        pt = predict(model, datasets["evaluation"], tasks, args.device, args.batch_size)
        np.savez_compressed(run_dir / f"seed_{seed}_predictions.npz", validation=pv, evaluation=pt)
        val_preds.append(pv)
        eval_preds.append(pt)
        del model
    path = finalize_predictions(datasets, val_preds, eval_preds, tasks, args.split_seed, run_dir)
    (run_dir / "timing.json").write_text(json.dumps(dict(feature_seconds=prepared - started,
        training_and_prediction_seconds=time.monotonic() - prepared)))
    return path


def finalize_predictions(datasets, val_preds, eval_preds, tasks, split_seed, run_dir):
    raw = np.mean(eval_preds, axis=0)
    pv = np.mean(val_preds, axis=0)
    np.random.seed(split_seed)
    adjusted = calibrate(pv, raw, datasets["validation"], datasets["evaluation"], tasks, run_dir)
    path = run_dir / "predictions.jsonl"
    tmp = run_dir / "predictions.partial.jsonl"
    with tmp.open("w") as f:
        for i, example in enumerate(datasets["evaluation"]):
            r = example["row"]
            row = {k: r[k] for k in ("task", "models", "id", "target", "correct")}
            row.update(raw=float(raw[i]), betasb=float(adjusted[i]),
                       per_seed=[float(p[i]) for p in eval_preds])
            f.write(json.dumps(row) + "\n")
    tmp.replace(path)
    return path


def finish_saved(bundle, args, run_dir):
    """Finish calibration/export from saved arrays; never import Torch or train.

    Older seed arrays have no IDs, so validate their exact row ordering against
    the saved feature ledger before interpreting their values.
    """
    datasets = {split: [dict(row=r) for r in bundle["rows"] if r["split"] == split]
                for split in ("validation", "evaluation")}
    seen = {split: [] for split in datasets}
    with (run_dir / "features.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["split"] in seen:
                seen[row["split"]].append(tuple(row[k] for k in ("task", "models", "id", "target")))
    for split, examples in datasets.items():
        expected = [tuple(e["row"][k] for k in ("task", "models", "id", "target")) for e in examples]
        if seen[split] != expected:
            raise ValueError(f"Saved {split} feature order/targets differ; cannot reuse prediction arrays")
    val_preds, eval_preds = [], []
    for seed in range(args.seeds):
        path = run_dir / f"seed_{seed}_predictions.npz"
        with np.load(path, allow_pickle=False) as saved:
            pv, pt = saved["validation"].copy(), saved["evaluation"].copy()
        for split, values in (("validation", pv), ("evaluation", pt)):
            if values.shape != (len(datasets[split]),) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"Invalid {split} predictions in {path}")
        val_preds.append(pv)
        eval_preds.append(pt)
    return finalize_predictions(datasets, val_preds, eval_preds, sorted(bundle["splits"]), args.split_seed, run_dir)
