"""New pooling rules; each returns fitting confidences, evaluation confidences and evaluation ranking scores."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax

from conf_compose.composition.candidates import EPSILON

# Penalties are on the summed fitting NLL; infinity means the equal-weight model exactly.
LOGIT_GRID = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, np.inf)
AVERAGE_GRID = (0.0, 10.0, 100.0, 1000.0, 10000.0, np.inf)
FOLDS = 5
SEED = 0
# Member log-odds are exact down to float64 resolution (at most 36.7 nats); the pipeline's 1e-3 clip would cap
# near-certain sequence probabilities at 6.9 nats and tie them. Consistency (1/12 to 11/12) never reaches either bound.
RESOLUTION = 2.0 ** -53


def logits(probabilities):
    clipped = np.clip(probabilities, RESOLUTION, 1 - RESOLUTION)
    return np.log(clipped) - np.log1p(-clipped)


def folds(labels):
    """Stratified fold per question, so every fold sees both outcomes."""
    rng = np.random.default_rng(SEED)
    index = np.empty(len(labels), dtype=int)
    for value in (0, 1):
        members = rng.permutation(np.flatnonzero(labels == value))
        index[members] = np.arange(len(members)) % FOLDS
    return index


def nll(y, z):
    return float(np.mean(np.logaddexp(0, z) - y * z))


def cross_validated(x, y, grid, fit, linear, one_se=False):
    """Penalty with the lowest out-of-fold NLL, or with one_se the strongest within one standard error of it."""
    if min(np.bincount(y.astype(int), minlength=2)) < FOLDS:
        return grid[-1]
    index = folds(y)
    losses = np.array([[nll(y[index == k], linear(fit(x[index != k], y[index != k], penalty), x[index == k]))
                        for k in range(FOLDS)] for penalty in grid])
    means = losses.mean(1)
    best = int(np.argmin(means))
    slack = losses[best].std(ddof=1) / np.sqrt(FOLDS) if one_se else 1e-9
    return max(p for p, loss in zip(grid, means) if loss <= means[best] + slack)


def logistic(design, y, ridge):
    """Logistic regression with a per-coefficient ridge; zero ridge entries are unpenalized."""
    def objective(theta):
        z = design @ theta
        return (np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(ridge * theta ** 2),
                design.T @ (expit(z) - y) + ridge * theta)

    start = np.zeros(design.shape[1])
    start[0] = np.log(max(y.mean(), 1e-3) / max(1 - y.mean(), 1e-3))
    return minimize(objective, start, jac=True, method="L-BFGS-B",
                    options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8}).x


def anchored_fit(x, y, penalty):
    """z = b + a * sum(logits) + sum(delta * logits); the ridge on delta pulls every weight to the shared a."""
    total = x.sum(1, keepdims=True)
    if np.isinf(penalty):
        b, a = logistic(np.column_stack([np.ones(len(x)), total]), y, np.array([0.0, 1e-6]))
        return b, np.full(x.shape[1], a)
    theta = logistic(np.column_stack([np.ones(len(x)), total, x]), y,
                     np.r_[0.0, 1e-6, np.full(x.shape[1], max(penalty, 1e-6))])
    return theta[0], theta[1] + theta[2:]


def nonnegative_fit(x, y, penalty):
    """z = b + sum(w * logits) with w >= 0; the ridge on w minus its mean pulls every weight to a shared value."""
    n = x.shape[1]
    if np.isinf(penalty):
        return anchored_fit(x, y, penalty)
    design = np.column_stack([np.ones(len(x)), x])
    center = np.eye(n) - 1 / n

    def objective(theta):
        z = design @ theta
        w = theta[1:]
        spread = center @ w
        return (np.sum(np.logaddexp(0, z) - y * z) + 0.5 * penalty * spread @ spread,
                design.T @ (expit(z) - y) + np.r_[0.0, penalty * center @ spread])

    start = np.r_[np.log(max(y.mean(), 1e-3) / max(1 - y.mean(), 1e-3)), np.full(n, 1 / n)]
    theta = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=[(None, None)] + [(0, None)] * n,
                     options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8}).x
    return theta[0], theta[1:]


def anchored_linear(parameters, x):
    b, w = parameters
    return b + x @ w


def weighted_sum(grid=LOGIT_GRID, nonnegative=False, one_se=False):
    """Learned log-odds weights that fall back to the (scaled) sum when the data cannot support more."""
    fit = nonnegative_fit if nonnegative else anchored_fit

    def rule(fit_p, fit_y, eval_p):
        x, e = logits(fit_p), logits(eval_p)
        parameters = fit(x, fit_y, cross_validated(x, fit_y, grid, fit, anchored_linear, one_se))
        z = anchored_linear(parameters, e)
        return expit(anchored_linear(parameters, x)), expit(z), z
    return rule


def average_fit(p, y, penalty):
    """Weighted average on the simplex with a logit link; the ridge pulls the weights to equal."""
    n = p.shape[1]
    uniform = np.full(n, 1 / n)

    def unpack(theta):
        return softmax(theta[:n]), theta[n], theta[n + 1]

    def objective(theta):
        w, a, b = unpack(theta)
        q = np.clip(p @ w, EPSILON, 1 - EPSILON)
        lq = np.log(q) - np.log1p(-q)
        z = a * lq + b
        r = expit(z) - y
        gap = w - uniform
        loss = np.sum(np.logaddexp(0, z) - y * z) + (penalty * gap @ gap if np.isfinite(penalty) else 0)
        dq = (r * a / (q * (1 - q))) @ p + (2 * penalty * gap if np.isfinite(penalty) else 0)
        dv = w * (dq - dq @ w)
        return loss, np.r_[dv, r @ lq, r.sum()]

    start = np.r_[np.zeros(n), 1.0, 0.0]
    if np.isinf(penalty):
        q = np.clip(p.mean(1), EPSILON, 1 - EPSILON)
        b, a = logistic(np.column_stack([np.ones(len(q)), np.log(q) - np.log1p(-q)]), y, np.array([0.0, 1e-6]))
        return uniform, a, b
    result = minimize(objective, start, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-12})
    return unpack(result.x)


def average_linear(parameters, p):
    w, a, b = parameters
    q = np.clip(p @ w, EPSILON, 1 - EPSILON)
    return a * (np.log(q) - np.log1p(-q)) + b


def weighted_average(one_se=False, grid=AVERAGE_GRID):
    def rule(fit_p, fit_y, eval_p):
        penalty = cross_validated(fit_p, fit_y, grid, average_fit, average_linear, one_se)
        parameters = average_fit(fit_p, fit_y, penalty)
        return (expit(average_linear(parameters, fit_p)), expit(average_linear(parameters, eval_p)),
                eval_p @ parameters[0])
    return rule


RULES = {
    "weighted_sum": weighted_sum(),
    "weighted_sum_1se": weighted_sum(one_se=True),
    "weighted_sum_nonneg": weighted_sum(nonnegative=True),
    "weighted_sum_nonneg_1se": weighted_sum(nonnegative=True, one_se=True),
    "weighted_average": weighted_average(),
    "weighted_average_1se": weighted_average(one_se=True),
}
