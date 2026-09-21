"""Regression tests for composition: the audit's counterexamples plus the pooling invariants."""

import math

import pytest

from conf_compose.composition import (OTHER, Item, Panel, Row, Stream, available, blocks, candidate_set,
                                      evaluate, fit_intercepts, fixed_answer_methods, label_space,
                                      majority_answer, pool_methods, prior_corrected, sources, state_space,
                                      support_of)


class NumericTask:
    """Open-answer stub where 12 and 12.0 are the same answer."""
    name = "numeric"

    def equivalent(self, a, b):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return str(a).strip() == str(b).strip()

    def is_correct(self, predicted, example):
        return predicted is not None and self.equivalent(predicted, example.answer)


class ChoiceTask(NumericTask):
    name = "csqa"

    def equivalent(self, a, b):
        return str(a).strip().upper() == str(b).strip().upper()


def stream(agent, answer, samples, **signals):
    return Stream(f"e:r0:a{agent}", agent, 0, f"m{agent}", answer, samples, tokens=10, signals=signals)


def test_binary_support_is_independent_of_unrelated_candidate_count():
    """Audit 1: the same fixed event must not change score because other answers appeared."""
    task = NumericTask()
    samples = ["7", "7", "7", "3", "9"]
    two = support_of(task, samples, ["7", "3"]).binary("7")
    four = support_of(task, samples, ["7", "3", "9", "11"]).binary("7")
    assert two == four == pytest.approx((3 + 0.5) / (5 + 1))


def test_equivalent_strings_keep_their_support():
    """Audit 2: counts stored under `12` must be found when the target is written `12.0`."""
    task = NumericTask()
    support = support_of(task, ["12", "12", "12"], ["12"])
    assert support.raw("12.0") == pytest.approx(1.0)
    assert support.binary("12.0") == pytest.approx(3.5 / 4)
    assert support.key("12.0") == "12"
    assert support.key("13") == OTHER


def test_paired_comparison_uses_each_method_own_labels():
    """Audit 3: a selector that picks a different answer is scored against its own correctness."""
    ids = [str(i) for i in range(20)]
    reference = Row("ref", ids, [i / 20 for i in range(20)], [0.0] * 10 + [1.0] * 10, True, ids, 20)
    other = Row("other", ids, [i / 20 for i in range(20)], [1.0] * 10 + [0.0] * 10, True, ids, 20)
    report = evaluate([reference, other], "ref", n_boot=20)
    assert report["other"]["auroc"] == pytest.approx(0.0)
    assert report["other"]["delta_auroc"] == pytest.approx(-1.0)


def test_length_is_log_scaled_and_calibration_uses_stored_logits():
    """Audit 4: length must not saturate a sigmoid, and pooled logits must survive into calibration."""
    task = NumericTask()
    item = Item("e1", "q", "7", [stream(0, "7", ["7"] * 5)])
    methods = fixed_answer_methods(task, item, item.streams, "7", target_tokens=400)
    assert methods["length"].score == pytest.approx(-math.log1p(400))
    assert methods["logodds_sum"].logit is not None
    row = Row("logodds_sum", ["a", "b"], [0.999999, 0.5], [1.0, 0.0], True, [None, None], 2,
              logits=[12.0, 0.0])
    assert fit_intercepts([row])["logodds_sum"] == pytest.approx(-6.0, abs=0.5)


def test_state_space_is_the_full_label_set_for_closed_tasks():
    """Closed-label tasks get every option as a probability state; open tasks keep a residual OTHER."""
    assert state_space(ChoiceTask(), ["A", "C"]) == list(label_space(ChoiceTask()))
    assert state_space(NumericTask(), ["7", "3"]) == ["7", "3", OTHER]


def test_blocks_and_infeasible_panels_produce_no_sources():
    """A homogeneous panel that needs more samples than exist must drop out, not silently shrink."""
    task = NumericTask()
    assert blocks(["a"] * 5, 2, 2) == [["a", "a"], ["a", "a"]]
    assert blocks(["a"] * 5, 4, 2) == []
    item = Item("e1", "q", "7", [stream(0, "7", ["7"] * 5, verification=0.9)])
    panel = Panel(("m0",), streams_per_model=4, samples_per_stream=2, families=("consistency", "verification"))
    assert sources(task, item, "7", ["7"], panel, item.streams) == []


def test_self_ratings_are_ineligible_for_another_answer():
    """Verification rates only its own model's answer; for a different target it must be dropped."""
    task = NumericTask()
    item = Item("e1", "q", "7", [stream(0, "3", ["3"] * 5, verification=0.9)])
    panel = Panel(("m0",), 1, 5, ("consistency", "verification"))
    built = sources(task, item, "7", ["7", "3"], panel, item.streams)
    assert [s.family for s in built if s.eligible] == ["consistency"]
    assert len(available(built)) == 1


def test_duplication_leaves_the_means_unchanged_but_sharpens_the_sum():
    """Controlled overcounting: duplicated evidence must not be read as fresh evidence by a mean rule."""
    single = pool_methods([0.8, 0.6])
    doubled = pool_methods([0.8, 0.6, 0.8, 0.6])
    assert doubled["mean"].score == pytest.approx(single["mean"].score)
    assert doubled["logodds_mean"].score == pytest.approx(single["logodds_mean"].score)
    assert doubled["logodds_sum"].score > single["logodds_sum"].score


def test_prior_cancels_exactly_at_the_mean_weight():
    """The reference prior drops out of the weighted rule when w = 1/S, for any prior."""
    logits = [1.2, -0.4, 0.7]
    for prior in (0.2, 0.5, 0.9):
        pooled = prior_corrected(logits, prior, w=1 / len(logits))
        assert pooled == pytest.approx(pool_methods([_sig(v) for v in logits])["logodds_mean"].score, abs=1e-6)


def test_majority_answer_respects_equivalence():
    task = NumericTask()
    streams = [stream(0, "12", []), stream(1, "12.0", []), stream(2, "9", [])]
    assert task.equivalent(majority_answer(task, streams), "12")


def _sig(value):
    return 1 / (1 + math.exp(-value))


def test_state_space_follows_the_example_when_option_counts_vary():
    """TruthfulQA questions carry their own option letters, so the state space is per example."""
    class VariableChoice(NumericTask):
        name = "truthfulqa"
    task = VariableChoice()
    assert state_space(task, ["A", "C"], list("ABCDEFG")) == list("ABCDEFG")
    assert state_space(task, ["A", "C"]) == ["A", "C", OTHER]


def test_choice_options_are_shuffled_off_the_first_position():
    """The sources list the correct answer first, so an unshuffled task would always be answered A."""
    from conf_compose.data.multiple_choice import _choice_example
    golds = {_choice_example("ABCD", f"q-{i}", "why?", ["right", "w1", "w2", "w3"], 0).answer
             for i in range(25)}
    assert len(golds) > 1


def test_choice_example_refuses_to_silently_drop_options():
    """More options than letters must fail loudly, not truncate the question."""
    from conf_compose.data.multiple_choice import _choice_example
    with pytest.raises(ValueError):
        _choice_example("ABC", "q-0", "why?", ["a", "b", "c", "d"], 0)
