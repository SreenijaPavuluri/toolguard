"""Metrics with bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def recall_at_fpr(y, s, fpr=0.01):
    y, s = np.asarray(y), np.asarray(s)
    neg = np.sort(s[y == 0])
    if len(neg) == 0 or (y == 1).sum() == 0:
        return float("nan")
    # threshold = smallest score such that at most `fpr` of negatives are strictly above it
    k = int(np.floor(fpr * len(neg)))
    thr = neg[len(neg) - 1 - k] if k < len(neg) else neg[0]
    return float((s[y == 1] > thr).mean())


def point_metrics(y, s, pred):
    y, s, pred = np.asarray(y), np.asarray(s), np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    auc = roc_auc_score(y, s) if len(set(y.tolist())) == 2 else float("nan")
    return {"precision": p, "recall": r, "f1": f1, "auroc": float(auc), "recall_at_1fpr": recall_at_fpr(y, s)}


def bootstrap(y, s, pred, n=1000, seed=0):
    y, s, pred = np.asarray(y), np.asarray(s), np.asarray(pred)
    rng = np.random.default_rng(seed)
    point = point_metrics(y, s, pred)
    samples = {k: [] for k in point}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        m = point_metrics(y[idx], s[idx], pred[idx])
        for k, v in m.items():
            samples[k].append(v)
    return {k: {"value": point[k], "lo": float(np.nanpercentile(samples[k], 2.5)),
                "hi": float(np.nanpercentile(samples[k], 97.5))} for k in point}


def rate_ci(x, n=1000, seed=0):
    """Mean of a 0/1 vector with a bootstrap 95% CI."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return {"value": float(x.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n": len(x)}
