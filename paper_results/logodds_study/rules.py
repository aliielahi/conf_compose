"""Log-odds pooling rules that learn how much evidence each confidence level carries; each returns
fitting confidences, evaluation confidences and evaluation ranking scores, like weighting_study/methods.py."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.special import expit

# The weighting study's building blocks, loaded by path so the two studies never shadow each other.
_spec = importlib.util.spec_from_file_location("weighting_methods", Path(__file__).resolve().parents[1]
                                               / "weighting_study" / "methods.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
LOGIT_GRID, anchored_fit, anchored_linear = _base.LOGIT_GRID, _base.anchored_fit, _base.anchored_linear
cross_validated, logistic, logits = _base.cross_validated, _base.logistic, _base.logits

# Ridge on the per-level corrections, summed over the fitting NLL; infinity is the fitted-scale sum exactly.
EVIDENCE_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, np.inf)
# Beta-prior strength used to turn k matching samples out of n into a probability; 0.5 is what the store uses.
ALPHA_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 1e6)
SAMPLES = 5


def knots(x):
    """Every logit level that holds at least 1% of the fitting entries, plus the extremes."""
    values, counts = np.unique(np.round(x, 9), return_counts=True)
    keep = values[counts >= 0.01 * x.size]
    return np.unique(np.r_[values.min(), keep, values.max()])


def hats(x, grid):
    """Per question, how much of each level's tent the members fall on; members between levels are split linearly."""
    x = np.clip(x, grid[0], grid[-1])
    upper = np.clip(np.searchsorted(grid, x, side="right"), 1, len(grid) - 1)
    lower = upper - 1
    share = (x - grid[lower]) / (grid[upper] - grid[lower])
    counts = np.zeros((x.shape[0], len(grid)))
    rows = np.repeat(np.arange(x.shape[0]), x.shape[1])
    np.add.at(counts, (rows, lower.ravel()), (1 - share).ravel())
    np.add.at(counts, (rows, upper.ravel()), share.ravel())
    return counts


def evidence_fit(x, y, penalty, grid):
    """z = b + a * sum(logits) + sum over members of a learned correction at each level, ridge-pulled to zero."""
    total = x.sum(1, keepdims=True)
    if np.isinf(penalty):
        b, a = logistic(np.column_stack([np.ones(len(x)), total]), y, np.array([0.0, 1e-6]))
        return b, a, np.zeros(len(grid))
    theta = logistic(np.column_stack([np.ones(len(x)), total, hats(x, grid)]), y,
                     np.r_[0.0, 1e-6, np.full(len(grid), penalty)])
    return theta[0], theta[1], theta[2:]


def evidence_linear(grid):
    def linear(parameters, x):
        b, a, delta = parameters
        return b + a * x.sum(1) + hats(x, grid) @ delta
    return linear


def evidence_sum(one_se=True, grid=EVIDENCE_GRID):
    """Sum of learned per-level log-likelihood ratios: each agent's stated log-odds is recalibrated by level."""
    def rule(fit_p, fit_y, eval_p):
        x, e = logits(fit_p), logits(eval_p)
        levels = knots(x)
        linear = evidence_linear(levels)

        def fit(xs, ys, penalty):
            return evidence_fit(xs, ys, penalty, levels)
        parameters = fit(x, fit_y, cross_validated(x, fit_y, grid, fit, linear, one_se))
        z = linear(parameters, e)
        return expit(linear(parameters, x)), expit(z), z
    return rule


def counts(p):
    """(matching samples, valid samples) behind each stored (k + 0.5) / (n + 1); five samples wins ties."""
    k, n = np.full(p.shape, np.nan), np.full(p.shape, np.nan)
    for samples in range(SAMPLES, 0, -1):
        guess = np.round(p * (samples + 1) - 0.5)
        hit = np.isnan(k) & np.isclose((guess + 0.5) / (samples + 1), p, atol=1e-6) & (guess >= 0) & (guess <= samples)
        k[hit], n[hit] = guess[hit], samples
    miss = np.isnan(k)
    k[miss], n[miss] = p[miss] * SAMPLES, SAMPLES
    return k, n


def smoothed_logits(p, alpha):
    k, n = counts(p)
    return np.log(k + alpha) - np.log(n - k + alpha)


def alpha_sum(one_se=False, grid=ALPHA_GRID):
    """Fitted-scale log-odds sum whose Beta-prior strength is chosen by cross-validation."""
    def rule(fit_p, fit_y, eval_p):
        def fit(p, y, alpha):
            return logistic(np.column_stack([np.ones(len(p)), smoothed_logits(p, alpha).sum(1)]), y,
                            np.array([0.0, 1e-6])), alpha

        def linear(parameters, p):
            (b, a), alpha = parameters
            return b + a * smoothed_logits(p, alpha).sum(1)
        parameters = fit(fit_p, fit_y, cross_validated(fit_p, fit_y, grid, fit, linear, one_se))
        z = linear(parameters, eval_p)
        return expit(linear(parameters, fit_p)), expit(z), z
    return rule


def round_fit(x, y, penalty, split):
    """z = b + a * sum(logits) + c * (post-debate sum - round-0 sum); the ridge pulls c to zero."""
    total = x.sum(1, keepdims=True)
    gap = (x[:, :split].sum(1) - x[:, split:].sum(1))[:, None]
    if np.isinf(penalty):
        b, a = logistic(np.column_stack([np.ones(len(x)), total]), y, np.array([0.0, 1e-6]))
        return b, a, 0.0
    return tuple(logistic(np.column_stack([np.ones(len(x)), total, gap]), y, np.array([0.0, 1e-6, penalty])))


def round_sum(split, one_se=True, grid=LOGIT_GRID):
    """Separate scales for post-debate and round-0 streams, which come from the same agents and overlap."""
    def linear(parameters, x):
        b, a, c = parameters
        return b + a * x.sum(1) + c * (x[:, :split].sum(1) - x[:, split:].sum(1))

    def rule(fit_p, fit_y, eval_p):
        x, e = logits(fit_p), logits(eval_p)

        def fit(xs, ys, penalty):
            return round_fit(xs, ys, penalty, split)
        parameters = fit(x, fit_y, cross_validated(x, fit_y, grid, fit, linear, one_se))
        z = linear(parameters, e)
        return expit(linear(parameters, x)), expit(z), z
    return rule


def evidence_weights(one_se=True):
    """Per-level evidence first, then the existing per-agent learned weights on the recalibrated logits."""
    def rule(fit_p, fit_y, eval_p):
        x, e = logits(fit_p), logits(eval_p)
        levels = knots(x)
        linear = evidence_linear(levels)

        def fit(xs, ys, penalty):
            return evidence_fit(xs, ys, penalty, levels)
        b, a, delta = fit(x, fit_y, cross_validated(x, fit_y, EVIDENCE_GRID, fit, linear, one_se))
        # Each member's recalibrated evidence, so the agent weights act on evidence rather than stated log-odds.
        member = lambda v: a * v + (hats(v.reshape(-1, 1), levels) @ delta).reshape(v.shape)  # noqa: E731
        fx, fe = member(x), member(e)
        parameters = anchored_fit(fx, fit_y, cross_validated(fx, fit_y, LOGIT_GRID, anchored_fit, anchored_linear, one_se))
        z = anchored_linear(parameters, fe)
        return expit(anchored_linear(parameters, fx)), expit(z), z
    return rule


def spread(x):
    """How far apart the members' log-odds are; zero when they all agree."""
    return x.std(1)


def dispersion_fit(x, y, penalty, agent_penalty):
    """z = b + a * sum(logits) + sum(delta * logits) + c * spread(logits); ridges pull delta and c to zero."""
    columns = [np.ones(len(x)), x.sum(1), spread(x)]
    ridge = [0.0, 1e-6, penalty if np.isfinite(penalty) else 1e12]
    if agent_penalty is not None:
        columns += list(x.T)
        ridge += [agent_penalty if np.isfinite(agent_penalty) else 1e12] * x.shape[1]
    theta = logistic(np.column_stack(columns), y, np.array(ridge, dtype=float))
    delta = theta[3:] if agent_penalty is not None else np.zeros(x.shape[1])
    return theta[0], theta[1] + delta, theta[2]


def dispersion_linear(parameters, x):
    b, w, c = parameters
    return b + x @ w + c * spread(x)


def dispersion_sum(one_se=True, agents=False, grid=LOGIT_GRID):
    """Log-odds sum with a disagreement term; with agents, also the per-agent learned weights (penalties chosen in turn)."""
    def rule(fit_p, fit_y, eval_p):
        x, e = logits(fit_p), logits(eval_p)
        penalty = cross_validated(x, fit_y, grid, lambda xs, ys, p: dispersion_fit(xs, ys, p, None),
                                  dispersion_linear, one_se)
        agent_penalty = None
        if agents:
            agent_penalty = cross_validated(x, fit_y, grid, lambda xs, ys, p: dispersion_fit(xs, ys, penalty, p),
                                            dispersion_linear, one_se)
        parameters = dispersion_fit(x, fit_y, penalty, agent_penalty)
        z = dispersion_linear(parameters, e)
        return expit(dispersion_linear(parameters, x)), expit(z), z
    return rule


def jensen_gap(x):
    """logit of the members' mean probability minus their mean log-odds: zero when they agree, grows as they split."""
    return logits(expit(x).mean(1)) - x.mean(1)


def gap_sum(one_se=True, agents=False, grid=LOGIT_GRID):
    """Log-odds sum corrected by the members' disagreement; with agents, also the per-agent learned weights."""
    def fit(x, y, penalty, agent_penalty):
        columns = [np.ones(len(x)), x.sum(1), jensen_gap(x)]
        ridge = [0.0, 1e-6, penalty if np.isfinite(penalty) else 1e12]
        if agent_penalty is not None:
            columns += list(x.T)
            ridge += [agent_penalty if np.isfinite(agent_penalty) else 1e12] * x.shape[1]
        theta = logistic(np.column_stack(columns), y, np.array(ridge, dtype=float))
        return theta[0], theta[1] + (theta[3:] if agent_penalty is not None else np.zeros(x.shape[1])), theta[2]

    def linear(parameters, x):
        b, w, c = parameters
        return b + x @ w + c * jensen_gap(x)

    def rule(fit_p, fit_y, eval_p):
        x, e = logits(fit_p), logits(eval_p)
        penalty = cross_validated(x, fit_y, grid, lambda xs, ys, p: fit(xs, ys, p, None), linear, one_se)
        agent_penalty = (cross_validated(x, fit_y, grid, lambda xs, ys, p: fit(xs, ys, penalty, p), linear, one_se)
                         if agents else None)
        parameters = fit(x, fit_y, penalty, agent_penalty)
        z = linear(parameters, e)
        return expit(linear(parameters, x)), expit(z), z
    return rule
