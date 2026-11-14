"""Helpers shared by the supplementary analyses written for the manuscript.

These scripts only read the project's code, data and saved outputs. Everything
they produce goes to paper/analysis/results/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import anomaly_core as core  # noqa: E402

RESULTS = ROOT / "paper" / "analysis" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "Dataset_elliptic++"
SEED = 42


def lgbm(seed: int, rounds: int):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=rounds, learning_rate=core.LEARNING_RATE, num_leaves=63,
        min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.4,
        reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)


def summarise(y, score, threshold=0.5) -> dict:
    """Threshold metrics plus threshold-free ranking metrics for one group of rows."""
    y = np.asarray(y).astype(int)
    m = core.evaluate(y, np.asarray(score), threshold)
    keep = ("precision", "recall", "f1", "fpr", "mcc", "pr_auc", "roc_auc", "tp", "fp", "fn", "tn")
    out = {k: m[k] for k in keep}
    out["n"], out["n_pos"] = int(len(y)), int(y.sum())
    if 0 < y.sum() < len(y):
        out["oracle_f1"] = core.evaluate(y, score, core.best_f1_threshold(y, score))["f1"]
    return out


def auc_or_nan(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def ap_or_nan(y, s):
    y = np.asarray(y)
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def save(name: str, obj) -> None:
    path = RESULTS / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path.with_suffix(".csv"), index=False)
    else:
        path.with_suffix(".json").write_text(json.dumps(obj, indent=2, default=float))
    print(f"[saved] {name}", flush=True)


def homophily(edges: np.ndarray, y: np.ndarray) -> dict:
    """Label agreement along edges whose two endpoints are both labelled."""
    a, b = y[edges[:, 0]], y[edges[:, 1]]
    both = (a != -1) & (b != -1)
    a, b = a[both], b[both]
    base = float((y == 1).sum() / (y != -1).sum())
    touching_illicit = (a == 1) | (b == 1)
    # For a labelled neighbour of an illicit node: how often is it illicit too?
    nbr_of_illicit = np.concatenate([b[a == 1], a[b == 1]])
    nbr_of_licit = np.concatenate([b[a == 0], a[b == 0]])
    return {"labelled_edges": int(both.sum()), "base_rate_illicit": base,
            "edges_illicit_illicit": int(((a == 1) & (b == 1)).sum()),
            "edges_mixed": int((a != b).sum()),
            "edges_licit_licit": int(((a == 0) & (b == 0)).sum()),
            "edges_touching_illicit": int(touching_illicit.sum()),
            "p_neighbour_illicit_given_illicit": float(nbr_of_illicit.mean()) if len(nbr_of_illicit) else float("nan"),
            "p_neighbour_illicit_given_licit": float(nbr_of_licit.mean()) if len(nbr_of_licit) else float("nan")}
