"""Analyses that need only the saved scores (outputs/<layer>/<layer>_verdicts.csv).

For every layer: how much F1 is lost by transferring the validation threshold
to the test period (achieved vs. best possible on test), calibration (Brier
score, expected calibration error), triage workload of the three-way verdict,
recall under a fixed review budget, and the same broken down by period.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from common import ROOT, ap_or_nan, auc_or_nan, core, save

LAYERS = {
    "transaction": ("outputs/transaction", "time_step", "illicit", "licit"),
    "transaction_no_graph": ("outputs/transaction_no_graph", "time_step", "illicit", "licit"),
    "wallet": ("outputs/wallet", "time_step", "illicit", "licit"),
    "wallet_no_graph": ("outputs/wallet_no_graph", "time_step", "illicit", "licit"),
    "network_sampled": ("outputs/network", "day", "attack", "benign"),
    "network_full": ("paper/analysis/results/network_full", "day", "attack", "benign"),
}


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def block(y, s, thr):
    m = core.evaluate(y, s, thr)
    best_thr = core.best_f1_threshold(y, s) if 0 < y.sum() < len(y) else float("nan")
    oracle = core.evaluate(y, s, best_thr)["f1"] if best_thr == best_thr else float("nan")
    return {"n": int(len(y)), "n_pos": int(y.sum()), "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"], "fpr": m["fpr"], "pr_auc": ap_or_nan(y, s),
            "roc_auc": auc_or_nan(y, s), "threshold": float(thr), "oracle_threshold": best_thr,
            "oracle_f1": oracle, "f1_lost_to_threshold": oracle - m["f1"]}


out = {}
for name, (folder, step_col, pos, neg) in LAYERS.items():
    folder = ROOT / folder
    if not (folder / "metrics.json").exists():
        print(f"[skip] {name}")
        continue
    meta = json.loads((folder / "metrics.json").read_text())
    entity = meta["entity"]
    df = pd.read_csv(folder / f"{entity}_verdicts.csv")
    test = df[(df["split"] == "test") & (df["true_label"] != "unlabelled")].copy()
    y = (test["true_label"] == pos).to_numpy().astype(int)
    s = test["anomaly_score"].to_numpy()
    thr, rev = meta["decision_threshold"], meta["review_threshold"]
    res = {"ensemble": block(y, s, thr), "members": {}, "periods": {}, "budget": {}}
    for member in meta["models"]:
        res["members"][member] = block(y, test[f"score_{member}"].to_numpy(),
                                       meta["test_per_model"][member]["threshold"])
    res["calibration"] = {"brier": float(np.mean((s - y) ** 2)), "ece_10_bins": ece(y, s),
                          "mean_score": float(s.mean()), "prevalence": float(y.mean())}
    flagged, review = s >= thr, (s >= rev) & (s < thr)
    res["triage"] = {
        "share_positive_verdict": float(flagged.mean()), "share_suspicious": float(review.mean()),
        "recall_positive_verdict": float(y[flagged].sum() / y.sum()),
        "recall_positive_or_suspicious": float(y[flagged | review].sum() / y.sum()),
        "precision_positive_verdict": float(y[flagged].mean()) if flagged.any() else float("nan"),
        "precision_suspicious_band": float(y[review].mean()) if review.any() else float("nan"),
        "suspicious_band_empty": bool(not review.any()), "decision_threshold": thr,
        "review_threshold": rev}
    order = np.argsort(-s)
    for frac in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
        k = max(1, int(round(frac * len(s))))
        top = order[:k]
        res["budget"][f"{frac:.3f}"] = {"k": k, "precision": float(y[top].mean()),
                                        "recall": float(y[top].sum() / y.sum())}
    steps = test[step_col].to_numpy()
    if name.startswith(("transaction", "wallet")):
        for label, (lo, hi) in {"35-42": (35, 42), "43-49": (43, 49)}.items():
            m = (steps >= lo) & (steps <= hi)
            res["periods"][label] = block(y[m], s[m], thr)
    out[name] = res
    e = res["ensemble"]
    print(f"{name}: F1 {e['f1']:.3f} oracle {e['oracle_f1']:.3f} AP {e['pr_auc']:.3f} "
          f"ECE {res['calibration']['ece_10_bins']:.4f}", flush=True)
save("score_analysis", out)
