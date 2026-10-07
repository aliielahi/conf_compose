"""Local checks on real round-0 records with a fake model: two rounds, scoring, resume, schema and parsing."""

import hashlib
import importlib.util
import tempfile
from collections import Counter
from pathlib import Path

from conf_compose.constants import ROOT
from conf_compose.data import get_task
from conf_compose.debate import (DebateSettings, generate_round, load_round, load_with_scores, round0_budget,
                                 score_candidates, write_settings)
from conf_compose.debate.prompts import PEER_LABELS

GROUP = ("q3-4bi", "l31-8bi", "g3-12i")
STYLES = ("explicit", "agree", "keep", "explicit", "silent")


class FakeGeneration:
    def __init__(self, text, logprobs=None):
        self.text, self.logprobs = text, logprobs
        self.finish_reason, self.error, self.ok = "stop", None, True
        self.input_tokens, self.output_tokens = 100, 20


class FakeLLM:
    """Deterministic in the prompt, so reruns are stable; counts calls so batching can be checked."""

    def __init__(self, task):
        self.task, self.execution, self.cache = task, "", None
        self.generate_calls, self.scored, self.scopes = 0, 0, []

    def _pick(self, messages, salt=""):
        return int(hashlib.sha256((repr(messages) + salt).encode()).hexdigest(), 16)

    def _answer(self, value):
        return {"csqa": "ABCDE"[value % 5], "gsm8k": str(value % 7 + 10)}[self.task]

    def _text(self, messages, salt=""):
        value = self._pick(messages, salt)
        style = STYLES[value % len(STYLES)]
        if style == "agree":
            return f"Reconsidering, I agree with {PEER_LABELS[value % 2]} after checking."
        if style == "keep":
            return "Their arguments miss a step, so I maintain my original answer."
        if style == "silent":
            return "Beta's answer is correct on reflection."
        return f"Step by step it follows. The answer is {self._answer(value)}."

    def render(self, system, messages):
        return "\n".join(message["content"] for message in messages)

    def encode(self, texts):
        return {"input_ids": [[0] * (len(text) // 4) for text in texts]}

    def generate(self, prompts, max_tokens=512, temperature=0.0, logprobs=False, **kwargs):
        self.generate_calls += 1
        self.scopes.append(self.execution)
        return [FakeGeneration(self._text(p), [-0.1] * 12) for p in prompts]

    def prompt(self, contexts, n=1, **kwargs):
        self.scopes.append(self.execution)
        return [[self._text(c, f"sample{i}") for i in range(n)] for c in contexts]

    def score(self, prompts, continuations, prefixes=None, system=None):
        self.scored += len(prompts)
        return [[-0.3 * (1 + self._pick([str(p), c]) % 5)] * max(1, len(c) // 3) for p, c in zip(prompts, continuations)]


def run(task_name, out, limit=12):
    budget = round0_budget(task_name, GROUP)
    settings = DebateSettings(task_name, GROUP, max_tokens=budget, limit=limit)
    write_settings(settings, out=out)
    llms = {}
    for round_index in (1, 2):
        for model in GROUP:
            llms[model] = FakeLLM(task_name)
            written = generate_round(llms[model], model, [settings], round_index, context=32768, out=out)
            assert written == {settings.name: limit}, written
            assert llms[model].generate_calls == 1, "one batched generation per task, round and budget"
            assert any(s.endswith(":answer") for s in llms[model].scopes), "revision has its own scope"
            assert any(s == f"debate:{task_name}:s70v0:r{round_index}" for s in llms[model].scopes)
    for model in GROUP:
        assert score_candidates(FakeLLM(task_name), model, [settings], through=2, out=out) == {settings.name: 3 * limit}
    return settings


def check(task_name, settings, out):
    task = get_task(task_name)
    round0 = load_round(settings, GROUP[0], 0)
    rows = load_with_scores(settings, GROUP[0], 2, through=2, out=out)
    row = next(iter(rows.values()))
    stored = next(iter(round0.values()))
    missing = [key for key in ("id", "prompt", "gold", "prediction", "correct", "response", "confidence",
                               "sampled_answers", "token_logprobs", "candidate_scores") if key not in row]
    assert not missing, f"debate record lacks store fields: {missing}"
    expected = {"seq_response", "seq_response_min", "seq_response_tail10", "seq_response_debiased",
                "consistency_t0.7", "consistency_t0.7_margin", "consistency_t0.7_entropy"}
    assert expected == set(row["confidence"]), set(row["confidence"]) ^ expected
    assert expected <= set(stored["confidence"]), "the same signal names as round 0"
    assert all(len(r["sampled_answers"]["consistency_t0.7"]) == 5 for r in rows.values())

    first = rows[row["id"]]
    assert first["messages"][0]["content"] == round0[row["id"]]["prompt"], "round 1 opens with round 0's prompt"
    own_round1 = load_round(settings, GROUP[0], 1, out=out)[row["id"]]
    assert first["messages"][1]["content"] == own_round1["response"], "round 2 replays the agent's own round 1"
    assert first["messages"][1]["role"] == "assistant" and first["messages"][2]["role"] == "user"
    assert "The answer is" in first["messages"][2]["content"], "the format instruction is restated"
    assert all(label in first["messages"][2]["content"] for label in first["peer_labels"])

    sources = Counter(r["answer_source"] for m in GROUP for r in load_round(settings, m, 1, out=out).values())
    assert {"explicit", "peer_reference", "kept_previous"} <= set(sources), sources

    # "Beta's answer is correct" must resolve to Beta's answer, never to option C read off "correct".
    silent = [r for m in GROUP for r in load_round(settings, m, 1, out=out).values()
              if r["response"].startswith("Beta's answer is correct")]
    for r in silent:
        beta = r["peer_order"][1]
        peer = load_round(settings, beta, 0)[r["id"]]["prediction"]
        assert r["answer_source"] == "peer_reference" and task.equivalent(r["prediction"], peer), r

    orders = Counter(tuple(r["peer_order"]) for r in load_round(settings, GROUP[0], 1, out=out).values())
    assert len(orders) == 2, f"peer order must vary across questions: {orders}"

    entry = first["candidate_scores"]
    answers = {c["answer"] for c in entry["candidates"]}
    proposed = {load_round(settings, m, r, out=out)[row["id"]]["prediction"] for m in GROUP for r in (0, 1, 2)} - {None}
    assert all(any(task.equivalent(a, p) for a in answers) for p in proposed), "pool holds every proposed answer"
    assert all(c["direct"] and c["null"] for c in entry["candidates"])

    # Voting's own confidence code runs unchanged on debate records.
    spec = importlib.util.spec_from_file_location("atomic", ROOT / "runs/experiment02-voting_composition/atomic.py")
    atomic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(atomic)
    args = type("Args", (), {"context": "direct", "seq_score": "norm_sum"})()
    scores = [atomic.sequence_score(task, r, r["prediction"], args) for r in rows.values() if r["prediction"]]
    assert scores and all(0 <= s <= 1 for s in scores if s is not None)
    print(f"{task_name:<6} answer sources {dict(sources)}  peer orders {len(orders)}  "
          f"candidates/question {len(entry['candidates'])}  seq scores ok {sum(s is not None for s in scores)}")


def main():
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        for task_name in ("csqa", "gsm8k"):
            settings = run(task_name, out)
            check(task_name, settings, out)
        tight = DebateSettings("csqa", GROUP, max_tokens=768, limit=4)
        written = generate_round(FakeLLM("csqa"), GROUP[0], [tight], 1, context=600, out=out)
        overflow = load_round(tight, GROUP[0], 1, out=out)
        assert all(r["error"] == "context_overflow" and r["prediction"] is None for r in overflow.values()), written
    print("\nall debate checks passed")


if __name__ == "__main__":
    main()
