"""Bayes-rule pooling variants: priors (constant, per-agent, answer-count), the clipping bound on member evidence,
bounded evidence and full answer-space products. Each rule takes fitting and evaluation views and returns fitting
confidences, evaluation confidences and evaluation ranking scores, like weighting_study/methods.py. Views are
{p: member confidences, y: labels, k: answer-space sizes, logp: per-question member log-probabilities over the answer
space with the selected answer first, exact: member log-odds computed in log space (sequence only)}."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

_spec = importlib.util.spec_from_file_location("weighting_methods", Path(__file__).resolve().parents[1]
                                               / "weighting_study" / "methods.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
# The pipeline's clip (at most 6.9 nats per member) and one that only guards against log(0) (27.6 nats).
EPSILON = 1e-3
TINY = 1e-12
# Largest member log-odds a float64 probability can express; caps the infinite evidence of a lone candidate.
CAP = float(np.log1p(-2.0 ** -53) - np.log(2.0 ** -53))
# Shape grids chosen by out-of-fold NLL on the fitting split; the last entry is the default when a class is too rare.
CLIP_GRID = (1e-1, 1e-2, 1e-4, 1e-6, 1e-9, 1e-12, 1e-3)
BOUND_GRID = (1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 20.0, np.inf)
TEMPERATURE_GRID = (0.25, 0.5, 2.0, 1.0)


def view(cell, split):
    """The fitting or evaluation questions of a cell as one record of arrays."""
    logp = cell[f"{split}_logp"]
    exact = (np.array([q[:, 0] - np.logaddexp.reduce(q[:, 1:], axis=1) if q.shape[1] > 1 else np.full(len(q), np.inf)
                       for q in logp]) if cell["estimator"] != "cons" else evidence(cell[split], 0.0))
    return {"p": cell[split], "y": cell[f"{split}_labels"], "logp": logp, "exact": np.clip(exact, -CAP, CAP),
            "k": np.array([q.shape[1] for q in logp], dtype=float)}


def subset(data, indices):
    return {"p": data["p"][indices], "y": data["y"][indices], "k": data["k"][indices], "exact": data["exact"][indices],
            "logp": [data["logp"][i] for i in indices]}


def evidence(p, epsilon=EPSILON):
    """Member log-odds with probabilities clipped to [epsilon, 1 - epsilon]."""
    p = np.clip(p, epsilon, 1 - epsilon)
    return np.log(p) - np.log1p(-p)


def answer_prior(k, epsilon=EPSILON):
    """Log-odds that one of k equally likely answers is right; a lone candidate is clipped like any confidence."""
    return evidence(1 / np.maximum(k, 1), epsilon)


def platt(x, y):
    return base.logistic(np.column_stack([np.ones(len(x)), x]), y, np.array([0.0, 1e-6]))


def learned_weights(x, y, e):
    """The current log-odds Learned Weights (one-SE ridge toward the fitted-scale sum) on given member evidence."""
    parameters = base.anchored_fit(x, y, base.cross_validated(x, y, base.LOGIT_GRID, base.anchored_fit,
                                                              base.anchored_linear, one_se=True))
    z = base.anchored_linear(parameters, e)
    return expit(base.anchored_linear(parameters, x)), expit(z), z


def chosen(fit, score, grid):
    """The shape value whose fitted-scale pooled score has the lowest out-of-fold NLL on the fitting split."""
    def fitter(indices, y, value):
        return platt(score(subset(fit, indices), value), y), value

    def linear(parameters, indices):
        (b, a), value = parameters
        return b + a * score(subset(fit, indices), value)
    return base.cross_validated(np.arange(len(fit["y"])), fit["y"], grid, fitter, linear, one_se=False)


def fitted_scale(score, grid):
    """Platt on a pooled score with its shape parameter chosen by cross-validation."""
    def rule(fit, ev):
        value = chosen(fit, score, grid)
        b, a = platt(score(fit, value), fit["y"])
        z = b + a * score(ev, value)
        return expit(b + a * score(fit, value)), expit(z), z
    return rule


# Priors.

def shifted(offset, epsilon=EPSILON):
    """Learned Weights on l_i - offset_i: a constant prior per member, which the free intercept absorbs."""
    def rule(fit, ev):
        shift = offset(fit)
        return learned_weights(evidence(fit["p"], epsilon) - shift, fit["y"], evidence(ev["p"], epsilon) - shift)
    return rule


def base_rate(fit):
    """Every member's prior is the fitting base rate of the group answer, split so N members add N-1 corrections."""
    n = fit["p"].shape[1]
    return np.full(n, (n - 1) / n * evidence(np.array([fit["y"].mean()]))[0])


def agent_rate(fit):
    """Each member's prior is the log-odds of its own mean stated confidence on the fitting split."""
    return evidence(fit["p"].mean(0))


def count_prior(epsilon=EPSILON):
    """Bayes rule with a per-question answer-count prior: l_i - (N-1)/N * logit(1/K_q), then Learned Weights."""
    def rule(fit, ev):
        n = fit["p"].shape[1]
        x = evidence(fit["p"], epsilon) - (n - 1) / n * answer_prior(fit["k"], epsilon)[:, None]
        e = evidence(ev["p"], epsilon) - (n - 1) / n * answer_prior(ev["k"], epsilon)[:, None]
        return learned_weights(x, fit["y"], e)
    return rule


def learned_count_prior(epsilon=EPSILON):
    """z = b + a * sum(l_i) + c * logit(1/K_q); the one-SE ridge on c falls back to no prior term at all."""
    def columns(data, prior):
        return np.column_stack([np.ones(len(data)), data[:, :-1].sum(1)] + ([data[:, -1]] if prior else []))

    def fitter(data, y, penalty):
        prior = np.isfinite(penalty)
        return base.logistic(columns(data, prior), y, np.r_[0.0, 1e-6, [penalty] if prior else []]), prior

    def linear(parameters, data):
        theta, prior = parameters
        return columns(data, prior) @ theta

    def rule(fit, ev):
        x = np.column_stack([evidence(fit["p"], epsilon), answer_prior(fit["k"], epsilon)])
        e = np.column_stack([evidence(ev["p"], epsilon), answer_prior(ev["k"], epsilon)])
        parameters = fitter(x, fit["y"], base.cross_validated(x, fit["y"], base.LOGIT_GRID, fitter, linear, one_se=True))
        return expit(linear(parameters, x)), expit(linear(parameters, e)), linear(parameters, e)
    return rule


# How much evidence one member may contribute.

def clip_score(data, epsilon):
    return evidence(data["p"], epsilon).sum(1)


def clip_weights(fit, ev):
    """Clipping bound chosen by cross-validation on the fitting split, then Learned Weights on that evidence."""
    epsilon = chosen(fit, clip_score, CLIP_GRID)
    return learned_weights(evidence(fit["p"], epsilon), fit["y"], evidence(ev["p"], epsilon))


def unclipped_weights(fit, ev):
    """Learned Weights on member log-odds clipped only against log(0)."""
    return learned_weights(evidence(fit["p"], TINY), fit["y"], evidence(ev["p"], TINY))


def exact_weights(fit, ev):
    """Learned Weights on each member's log-odds computed from its log-probabilities, bounded only where float64 is."""
    return learned_weights(fit["exact"], fit["y"], ev["exact"])


def exact_sum(fit, ev):
    """Fitted-scale sum of the exact member log-odds: Bayes rule with one learned scale and no per-member weights."""
    b, a = platt(fit["exact"].sum(1), fit["y"])
    z = b + a * ev["exact"].sum(1)
    return expit(b + a * fit["exact"].sum(1)), expit(z), z


def bounded(x, bound):
    return x if np.isinf(bound) else bound * np.tanh(x / bound)


def bound_score(data, bound):
    return bounded(evidence(data["p"], TINY), bound).sum(1)


# Full answer-space product.

def product_score(logp, temperature=1.0, epsilon=EPSILON):
    """Log-odds of the target (column 0) under the normalized product of every member's answer distribution."""
    out = np.empty(len(logp))
    for i, member in enumerate(logp):
        # A lone candidate is scored as if every member left epsilon on one alternative.
        member = np.column_stack([member, np.full(len(member), -np.inf)]) if member.shape[1] == 1 else member
        logs = temperature * np.maximum(member, np.log(epsilon)).sum(0)
        out[i] = logs[0] - logsumexp(logs[1:])
    return out


def probability_weights(fit, ev):
    return base.weighted_average(one_se=True)(fit["p"], fit["y"], ev["p"])


# The probability-space rule with its logit link clipped only against log(0), so both spaces get the same clip.
_unclipped = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_unclipped)
_unclipped.EPSILON = TINY


def set_seed(seed):
    """Cross-validation fold seed for every rule."""
    base.SEED = _unclipped.SEED = seed


def unclipped_probability_weights(fit, ev):
    return _unclipped.weighted_average(one_se=True)(fit["p"], fit["y"], ev["p"])


def clip_at(epsilon):
    """Learned Weights on member log-odds clipped at a fixed epsilon (sensitivity of the unclipped rule)."""
    def rule(fit, ev):
        return learned_weights(evidence(fit["p"], epsilon), fit["y"], evidence(ev["p"], epsilon))
    return rule


def logodds_weights(fit, ev):
    return base.weighted_sum(one_se=True)(fit["p"], fit["y"], ev["p"])


RULES = {
    "weighted_average_1se": probability_weights,
    "weighted_sum_1se": logodds_weights,
    "prior_base_rate": shifted(base_rate),
    "prior_agent_rate": shifted(agent_rate),
    "prior_count": count_prior(),
    "prior_count_learned": learned_count_prior(),
    "prior_count_unclipped": count_prior(TINY),
    "prior_count_learned_unclipped": learned_count_prior(TINY),
    "clip_sum": fitted_scale(clip_score, CLIP_GRID),
    "clip_weights": clip_weights,
    "unclipped_weights": unclipped_weights,
    "bound_sum": fitted_scale(bound_score, BOUND_GRID),
    "product": fitted_scale(lambda data, t: product_score(data["logp"], t), (1.0,)),
    "product_tempered": fitted_scale(lambda data, t: product_score(data["logp"], t), TEMPERATURE_GRID),
    "product_unclipped": fitted_scale(lambda data, t: product_score(data["logp"], t, TINY), (1.0,)),
    "weighted_average_unclipped": unclipped_probability_weights,
    "clip_1e-6_weights": clip_at(1e-6),
    "clip_1e-9_weights": clip_at(1e-9),
    "clip_1e-15_weights": clip_at(1e-15),
    "exact_weights": exact_weights,
    "exact_sum": exact_sum,
}
