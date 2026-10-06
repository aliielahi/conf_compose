"""Mock-driven checks: one content generation per example, both views, all three confidence readings."""

import math

from conf_compose.data import get_task
from conf_compose.judge_for_conf import CONFIDENCE_MODES, Judge, JudgeConfig, PanelEntry, PanelView

TASK = get_task("csqa")


class FakeGeneration:
    def __init__(self, text, top_logprobs=None, finish_reason="stop"):
        self.text = text
        self.top_logprobs = top_logprobs
        self.finish_reason = finish_reason
        self.input_tokens = 100
        self.output_tokens = 20
        self.ok = True


VERDICT = ("All three models agree and their reasoning is sound.\n"
           "Justification: three independent chains reach the same elimination step.\n"
           "Confidence: 8")


class FakeLLM:
    """Separates content generations from one-token probes, so the one-generation rule is checked."""

    def __init__(self, text=VERDICT):
        self.text = text
        self.generations = 0
        self.probes = 0
        self.score_calls = 0
        self.scored = 0
        self.prompts = []

    def generate(self, prompts, max_tokens=512, **kwargs):
        if max_tokens == 1:
            self.probes += len(prompts)
            top = {"true": math.log(0.75), "false": math.log(0.25)}
            return [FakeGeneration("true", [top]) for _ in prompts]
        self.generations += len(prompts)
        self.prompts.extend(prompts)
        return [FakeGeneration(self.text) for _ in prompts]

    def score(self, prompts, continuations, prefixes=None, system=None):
        self.score_calls += 1
        self.scored += len(prompts)
        return [[-0.5 * (ord(c[0]) - 64)] for c in continuations]


def views(n=12):
    out = []
    for i in range(n):
        letters = ["A", "B", "B"] if i % 2 else ["C", "C", "D"]
        entries = [PanelEntry(f"model{j}", letters[j], f"reasoning from model {j}", 0.4 + 0.2 * j)
                   for j in range(3)]
        out.append(PanelView(f"csqa-test-{i}", f"question {i}?", "B", entries, "B" if i % 2 else "C"))
    return out


def check(view_name, modes):
    llm = FakeLLM()
    config = JudgeConfig(view=view_name, confidence_method="consistency_t0.7", modes=modes)
    verdicts = Judge(llm, TASK, config).run(views())

    assert llm.generations == 12, f"expected one content generation each, got {llm.generations}"
    assert len(verdicts) == 12
    assert all(v.final_answer in ("B", "C") for v in verdicts), "the given answer must be carried through"

    assert all(v.justification for v in verdicts), "a justification must be saved for every example"

    prompt = llm.prompts[0]
    assert "majority vote" in prompt and "Model A" in prompt
    assert "Justification:" in prompt, "the prompt must ask for a justification"
    assert "model0" not in prompt, "panel model names must stay anonymous"
    assert ("confidence in its own answer (consistency_t0.7)" in prompt) == (view_name == "reasoning_confidence")

    counts = {mode: sum(getattr(v, mode) is not None for v in verdicts) for mode in CONFIDENCE_MODES}
    for mode in CONFIDENCE_MODES:
        expected = 12 if mode in modes else 0
        assert counts[mode] == expected, f"{mode}: got {counts[mode]}, expected {expected}"
    print(f"{view_name:<22}{'+'.join(modes):<32}gen={llm.generations:<4}probes={llm.probes:<4}"
          f"score_calls={llm.score_calls:<3}scored={llm.scored:<4}"
          + " ".join(f"{m}={counts[m]}" for m in CONFIDENCE_MODES))
    return verdicts


def main():
    print(f"{'view':<22}{'modes':<32}cost and coverage")
    for name in ("reasoning", "reasoning_confidence"):
        check(name, CONFIDENCE_MODES)

    verdicts = check("reasoning_confidence", ("verbalized",))
    assert all(v.verbalized == 0.8 for v in verdicts), "Confidence: 8 must parse to 0.8"
    assert all(v.justification == "three independent chains reach the same elimination step."
               for v in verdicts), verdicts[0].justification

    # The justification is saved and round-trips, but no confidence is derived from it.
    row = verdicts[0].to_dict()
    assert row["justification"] == verdicts[0].justification and "justification" in row

    # An unlabelled verdict still records something: the last line before the rating.
    plain = Judge(FakeLLM("The panel is split and model B contradicts itself.\nConfidence: 3"),
                  TASK, JudgeConfig(modes=("verbalized",))).run(views(2))
    assert all(v.justification == "The panel is split and model B contradicts itself." for v in plain)
    assert all(v.verbalized == 0.3 for v in plain)

    # A bare rating leaves it empty rather than inventing one.
    bare = Judge(FakeLLM("Confidence: 5"), TASK, JudgeConfig(modes=("verbalized",))).run(views(2))
    assert all(v.justification == "" and v.verbalized == 0.5 for v in bare)

    verdicts = check("reasoning_confidence", ("ptrue",))
    assert all(abs(v.ptrue - 0.75) < 1e-9 for v in verdicts), "P(True) must be 0.75"

    verdicts = check("reasoning_confidence", ("seqprob",))
    for verdict in verdicts:
        assert 0 < verdict.seqprob <= 1, "seqprob must be a normalised probability"
        assert len(verdict.candidates) == 2, "the pool is the panel's distinct answers"
        assert all(c["logprobs"] for c in verdict.candidates)

    # Views must differ only in what is shown, never in the answer being rated.
    for a, b in zip(check("reasoning", ("verbalized",)), check("reasoning_confidence", ("verbalized",))):
        assert a.final_answer == b.final_answer

    # A one-answer panel leaves nothing to normalise over: certain, not a crash.
    one = [PanelView("csqa-test-0", "q?", "B", [PanelEntry("model0", "B", "r", 0.5)], "B")]
    assert Judge(FakeLLM(), TASK, JudgeConfig(modes=("seqprob",))).run(one)[0].seqprob == 1.0

    # No majority answer means no confidence to report.
    none = [PanelView("csqa-test-1", "q?", "B", [PanelEntry("model0", None, "r", None)], None)]
    verdict = Judge(FakeLLM(), TASK, JudgeConfig()).run(none)[0]
    assert verdict.ptrue is None and verdict.seqprob is None

    # The judge's own rating is the last labelled one, not a member confidence it quoted on the way.
    quoting = ("Model A's confidence: 0.9 and Model B answered with confidence 0.95.\n"
               "Justification: only one chain is sound.\n**Confidence:** 6")
    verdict = Judge(FakeLLM(quoting), TASK, JudgeConfig(modes=("verbalized",))).run(views(1))[0]
    assert verdict.verbalized == 0.6, verdict.verbalized
    assert verdict.justification == "only one chain is sound."

    # finish_reason and token counts are saved, so truncation is never invisible again.
    row = Judge(FakeLLM(), TASK, JudgeConfig(modes=("verbalized",))).run(views(1))[0].to_dict()
    assert row["finish_reason"] == "stop" and row["prompt_tokens"] == 100 and row["output_tokens"] == 20

    # A prompt that cannot fit the window is refused before any generation, never truncated.
    class Tight(FakeLLM):
        def render(self, system, messages):
            return messages[0]["content"]

        def encode(self, texts):
            return {"input_ids": [[0] * len(text) for text in texts]}

    tight = Tight()
    try:
        Judge(tight, TASK, JudgeConfig(modes=("verbalized",), context=600, max_tokens=512)).run(views(2))
    except ValueError as error:
        assert "exceed 88 tokens" in str(error), error
        assert tight.generations == 0, "nothing may be generated once a prompt is known to overflow"
    else:
        raise AssertionError("an overflowing prompt must be refused")
    roomy = Judge(Tight(), TASK, JudgeConfig(modes=("verbalized",), context=32768)).run(views(2))
    assert all(v.verbalized == 0.8 for v in roomy)

    # Missing confidences are shown as unavailable rather than silently dropped.
    llm = FakeLLM()
    gap = [PanelView("csqa-test-2", "q?", "B", [PanelEntry("model0", "B", "r", None)], "B")]
    Judge(llm, TASK, JudgeConfig(modes=("verbalized",))).run(gap)
    assert "unavailable" in llm.prompts[0]

    assert Judge(FakeLLM(), TASK, JudgeConfig()).run([]) == []

    # Answers-only is not a view here, and an unknown reading is refused rather than ignored.
    for kwargs in ({"view": "answer"}, {"modes": ("logit",)}):
        try:
            JudgeConfig(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"JudgeConfig({kwargs}) should have been rejected")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
