"""
Shared machinery for the three anomaly-detection layers (transaction, wallet,
network traffic): feature helpers, models, baselines, the temporal
fit / validate / refit / test protocol, and research-style reporting
(tables as CSV + LaTeX + Markdown, figures as PNG + PDF).

Protocol (identical for every layer)
------------------------------------
1. fit   : early time steps        -> train every model
2. val   : last training steps     -> early stopping, decision thresholds
3. refit : fit + val               -> final models
4. test  : later, unseen steps     -> reported once, never used for tuning

Every scored entity ends with a three-way verdict:
    positive   score >= decision threshold (maximises F1 on validation)
    suspicious score >= review threshold   (keeps >= 95% validation recall)
    negative   otherwise
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (HistGradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import QuantileTransformer

try:
    import lightgbm as lgb
except ImportError:  # optional
    lgb = None
try:
    import xgboost as xgb
except ImportError:  # optional
    xgb = None

log = logging.getLogger("anomaly")

ILLICIT, LICIT, UNKNOWN = 1, 0, -1
MAX_ROUNDS = 2000
PATIENCE = 150
LEARNING_RATE = 0.05
RF_PARAMS = {"n_estimators": 600, "min_samples_leaf": 1}

MODEL_LABELS = {
    "logreg": "Logistic Regression", "iforest": "Isolation Forest (unsupervised)",
    "rf_default": "Random Forest (50 trees, default)",
    "lgbm": "LightGBM", "xgb": "XGBoost", "rf": "Random Forest", "hgb": "HistGradientBoosting",
    "ensemble": "Ensemble (ours)",
}
BASELINES = ("logreg", "iforest", "rf_default")
# Baselines that keep the fixed 0.5 cut-off used in prior work instead of a tuned threshold.
FIXED_THRESHOLD = {"rf_default": 0.5}


# --------------------------------------------------------------------------- #
# Small data helpers
# --------------------------------------------------------------------------- #
def find_column(df: pd.DataFrame, candidates: set[str]) -> str:
    for col in df.columns:
        if col.strip().lower().replace("_", " ") in candidates:
            return col
    raise KeyError(f"None of {sorted(candidates)} found in columns {list(df.columns)[:8]}...")


def parse_classes(raw: pd.Series) -> np.ndarray:
    """Elliptic++ uses 1=illicit, 2=licit, 3=unknown (original Elliptic: 'unknown')."""
    mapping = {"1": ILLICIT, "2": LICIT, "3": UNKNOWN, "unknown": UNKNOWN,
               "illicit": ILLICIT, "licit": LICIT}
    as_str = raw.astype(str).str.strip().str.lower().str.replace(r"\.0$", "", regex=True)
    mapped = as_str.map(mapping)
    if mapped.isna().any():
        raise ValueError(f"Unexpected class labels: {sorted(as_str[mapped.isna()].unique())[:5]}")
    return mapped.to_numpy(dtype=np.int8)


def impute_and_clean(X: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """Replace inf/NaN with medians computed on the training period only."""
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(X[fit_mask], axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    rows, cols = np.nonzero(np.isnan(X))
    X[rows, cols] = med[cols]
    return X


def build_graph_features(X: np.ndarray, edges: np.ndarray, names: list[str]):
    """Append structural features and the mean feature vector of in-/out-neighbours.

    Only features are propagated, never labels. Nodes without neighbours in a
    direction get NaN for that block (tree models treat it as "missing").
    The original features stay in the first len(names) columns.
    """
    n = X.shape[0]
    A = sp.csr_matrix(
        (np.ones(len(edges), dtype=np.float32), (edges[:, 0], edges[:, 1])), shape=(n, n))
    AT = A.T.tocsr()
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    in_deg = np.asarray(AT.sum(axis=1)).ravel()
    deg = in_deg + out_deg

    def neighbour_mean(M, d):
        agg = np.asarray(M @ X, dtype=np.float32)
        agg /= np.maximum(d, 1.0)[:, None].astype(np.float32)
        agg[d == 0] = np.nan
        return agg

    _, comp = connected_components(A, directed=False)
    comp_size = np.bincount(comp)[comp]
    structural = np.column_stack([
        in_deg, out_deg, deg, out_deg / (deg + 1.0),
        AT @ in_deg, A @ out_deg,                      # 2-hop walk counts
        (AT @ deg) / np.maximum(in_deg, 1.0), (A @ deg) / np.maximum(out_deg, 1.0),
        np.log1p(comp_size),
    ]).astype(np.float32)
    structural_names = ["g_in_deg", "g_out_deg", "g_deg", "g_out_ratio", "g_in_2hop",
                        "g_out_2hop", "g_in_nbr_avg_deg", "g_out_nbr_avg_deg",
                        "g_log_component_size"]
    all_names = (list(names) + structural_names
                 + [f"in_nbr_mean__{c}" for c in names] + [f"out_nbr_mean__{c}" for c in names])
    return np.hstack([X, structural, neighbour_mean(AT, in_deg), neighbour_mean(A, out_deg)]), all_names


# --------------------------------------------------------------------------- #
# Models.  Trainer: (Xtr, ytr, Xva, yva, seed, n_rounds) -> (model, rounds_used)
# n_rounds=None -> early stopping on (Xva, yva); otherwise train exactly n_rounds.
# --------------------------------------------------------------------------- #
def train_lgbm(Xtr, ytr, Xva, yva, seed, n_rounds=None):
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=n_rounds or MAX_ROUNDS, learning_rate=LEARNING_RATE,
        num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.4, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
    if n_rounds is None:
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        return model, int(model.best_iteration_ or MAX_ROUNDS)
    model.fit(Xtr, ytr)
    return model, n_rounds


def train_xgb(Xtr, ytr, Xva, yva, seed, n_rounds=None):
    params = dict(objective="binary:logistic", tree_method="hist", learning_rate=LEARNING_RATE,
                  max_depth=8, subsample=0.8, colsample_bytree=0.4, reg_lambda=1.0,
                  eval_metric="aucpr", random_state=seed, n_jobs=-1)
    if n_rounds is None:
        model = xgb.XGBClassifier(n_estimators=MAX_ROUNDS, early_stopping_rounds=PATIENCE, **params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return model, int(model.best_iteration) + 1
    model = xgb.XGBClassifier(n_estimators=n_rounds, **params)
    model.fit(Xtr, ytr)
    return model, n_rounds


def train_hgb(Xtr, ytr, Xva, yva, seed, n_rounds=None):
    """scikit-learn fallback when neither LightGBM nor XGBoost is installed."""
    model = HistGradientBoostingClassifier(
        learning_rate=LEARNING_RATE, max_iter=n_rounds or 600, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=False, random_state=seed)
    model.fit(Xtr, ytr)
    if n_rounds is not None:
        return model, n_rounds
    ap = [average_precision_score(yva, p[:, 1]) for p in model.staged_predict_proba(Xva)]
    return model, int(np.argmax(ap)) + 1


def train_rf(Xtr, ytr, Xva, yva, seed, n_rounds=None):
    model = RandomForestClassifier(max_features="sqrt", class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=seed, **RF_PARAMS)
    return model.fit(Xtr, ytr), 0


def train_logreg(Xtr, ytr, Xva, yva, seed, n_rounds=None):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        QuantileTransformer(n_quantiles=200, output_distribution="normal",
                            subsample=100_000, random_state=seed),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))
    return model.fit(Xtr, ytr), 0


def train_iforest(Xtr, ytr, Xva, yva, seed, n_rounds=None):
    """Unsupervised: sees only the negative (normal) class, no labels of the positive class."""
    normal = Xtr[ytr == LICIT]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        IsolationForest(n_estimators=300, max_samples=min(4096, len(normal)),
                        random_state=seed, n_jobs=-1))
    return model.fit(normal), 0


def train_rf_default(Xtr, ytr, Xva, yva, seed, n_rounds=None):
    """Configuration reported by Elmougy & Liu (KDD 2023): default RF with 50 trees."""
    return RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=seed).fit(Xtr, ytr), 0


TRAINERS = {"lgbm": train_lgbm, "xgb": train_xgb, "rf": train_rf, "hgb": train_hgb,
            "logreg": train_logreg, "iforest": train_iforest, "rf_default": train_rf_default}


def available_models(requested: list[str]) -> list[str]:
    missing = {"lgbm": lgb is None, "xgb": xgb is None}
    names = []
    for name in requested:
        if missing.get(name):
            log.warning("%s is not installed - skipping", name)
        else:
            names.append(name)
    if not ({"lgbm", "xgb", "hgb"} & set(names)):
        log.warning("No boosting library available - adding sklearn HistGradientBoosting")
        names.append("hgb")
    return names


def _score_batch(model, X, rounds):
    if isinstance(model, Pipeline) and isinstance(model[-1], IsolationForest):
        return -model.score_samples(X)          # higher = more anomalous
    if rounds and isinstance(model, HistGradientBoostingClassifier):
        for i, p in enumerate(model.staged_predict_proba(X), start=1):
            if i == rounds:
                return p[:, 1]
    return model.predict_proba(X)[:, 1]


def proba(model, X, rounds: int = 0, batch: int = 200_000) -> np.ndarray:
    """Anomaly score (P(positive) for classifiers), computed in batches to bound memory."""
    return np.concatenate([_score_batch(model, X[i:i + batch], rounds)
                           for i in range(0, len(X), batch)])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def best_f1_threshold(y_true, score) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def recall_threshold(y_true, score, target: float) -> float:
    """Highest threshold that still recalls at least `target` of the positives."""
    pos = np.sort(score[y_true == 1])
    k = int(np.floor((1.0 - target) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def _counts_to_metrics(tp, fp, fn, tn) -> dict:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    denom = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    neg_f1 = 2 * tn / max(2 * tn + fn + fp, 1)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "fpr": float(fp / max(fp + tn, 1)),
            "mcc": float((tp * tn - fp * fn) / denom) if denom else 0.0,
            "micro_f1": float((tp + tn) / max(tp + fp + fn + tn, 1)),
            "macro_f1": float((f1 + neg_f1) / 2)}


def evaluate(y_true, score, threshold: float) -> dict:
    pred = score >= threshold
    pos = y_true == 1
    tp, fp = int((pred & pos).sum()), int((pred & ~pos).sum())
    fn, tn = int((~pred & pos).sum()), int((~pred & ~pos).sum())
    both = 0 < pos.sum() < len(pos)
    return {**_counts_to_metrics(tp, fp, fn, tn),
            "pr_auc": float(average_precision_score(y_true, score)) if both else float("nan"),
            "roc_auc": float(roc_auc_score(y_true, score)) if both else float("nan"),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "threshold": float(threshold)}


def bootstrap_ci(y_true, score, threshold, n_boot: int, seed: int) -> dict:
    """95% percentile bootstrap intervals over the test set."""
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(y_true)
    pos, pred = y_true == 1, score >= threshold
    n_auc = min(n_boot, 200 if n <= 300_000 else 50)
    rows = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        p, d = pos[idx], pred[idx]
        m = _counts_to_metrics(int((d & p).sum()), int((d & ~p).sum()),
                               int((~d & p).sum()), int((~d & ~p).sum()))
        if b < n_auc and 0 < p.sum() < n:
            m["pr_auc"] = average_precision_score(p, score[idx])
            m["roc_auc"] = roc_auc_score(p, score[idx])
        rows.append(m)
    frame = pd.DataFrame(rows)
    return {k: [float(frame[k].quantile(0.025)), float(frame[k].quantile(0.975))]
            for k in ("precision", "recall", "f1", "fpr", "mcc", "pr_auc", "roc_auc")
            if k in frame}


def per_time_step(y, score, steps, threshold) -> pd.DataFrame:
    rows = []
    for t in np.unique(steps):
        m = steps == t
        e = evaluate(y[m], score[m], threshold)
        rows.append({"time_step": int(t), "n_labelled": int(m.sum()),
                     "n_positive": int(y[m].sum()), "n_flagged": e["tp"] + e["fp"],
                     "tp": e["tp"], "fp": e["fp"], "fn": e["fn"],
                     "precision": e["precision"], "recall": e["recall"], "f1": e["f1"]})
    return pd.DataFrame(rows)


def feature_importance(models: dict, names: list[str]) -> pd.DataFrame | None:
    """Mean of per-model importances (ensemble members only), each normalised to sum to 1."""
    parts = []
    for name, fitted in models.items():
        for model in fitted:
            if name == "lgbm":
                imp = model.booster_.feature_importance(importance_type="gain")
            elif hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
            else:
                continue
            imp = np.asarray(imp, dtype=np.float64)
            if len(imp) == len(names) and imp.sum() > 0:
                parts.append(imp / imp.sum())
    if not parts:
        return None
    return (pd.DataFrame({"feature": names, "importance": np.mean(parts, axis=0)})
            .sort_values("importance", ascending=False, ignore_index=True))


# --------------------------------------------------------------------------- #
# Tables (CSV + LaTeX + Markdown)
# --------------------------------------------------------------------------- #
def _fmt(v, digits=3):
    if isinstance(v, (float, np.floating)):
        return "--" if np.isnan(v) else f"{v:.{digits}f}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return str(v)


def to_markdown(df: pd.DataFrame, digits=3) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in df.columns) + "|"
    body = ["| " + " | ".join(_fmt(v, digits) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def to_latex(df: pd.DataFrame, caption: str, label: str, digits=3) -> str:
    esc = lambda s: (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_")
                     .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#"))
    align = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in df.columns)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{esc(caption)}}}",
             rf"\label{{{label}}}", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(_fmt(v, digits)) for v in row) + r" \\"
              for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def write_table(df: pd.DataFrame, stem: Path, caption: str, label: str, digits=3) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".tex").write_text(to_latex(df, caption, label, digits), encoding="utf-8")
    stem.with_suffix(".md").write_text(f"**{caption}**\n\n{to_markdown(df, digits)}\n",
                                       encoding="utf-8")


# --------------------------------------------------------------------------- #
# Figures (PNG at 300 dpi + vector PDF)
# --------------------------------------------------------------------------- #
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MODEL_STYLE = {
    "ensemble": dict(color=INK, lw=2.4, ls="-", zorder=5),
    "lgbm": dict(color=BLUE, lw=1.6, ls="-"), "xgb": dict(color=ORANGE, lw=1.6, ls="-"),
    "rf": dict(color=AQUA, lw=1.6, ls="-"), "hgb": dict(color=YELLOW, lw=1.6, ls="-"),
    "logreg": dict(color=MUTED, lw=1.4, ls="--"), "iforest": dict(color=MUTED, lw=1.4, ls=":"),
    "rf_default": dict(color=MUTED, lw=1.4, ls="-."),
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": MUTED,
        "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 8,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })
    return plt


def _save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    import matplotlib.pyplot as plt
    plt.close(fig)


def make_figures(fig_dir: Path, *, y_test, test_scores: dict, ensemble_test, threshold,
                 review_threshold, by_step, importance, test_metrics, entity, positive,
                 negative, step_name) -> None:
    plt = _plt()
    title = entity.capitalize()
    order = [m for m in MODEL_STYLE if m in test_scores and m != "ensemble"]

    # 1-2. Precision-recall and ROC curves, all models on one axis each.
    for kind in ("pr", "roc"):
        fig, ax = plt.subplots(figsize=(7.4, 4.0))
        for name in order + ["ensemble"]:
            s = ensemble_test if name == "ensemble" else test_scores[name]
            if kind == "pr":
                p, r, _ = precision_recall_curve(y_test, s)
                x, yv, auc = r, p, average_precision_score(y_test, s)
            else:
                x, yv, _ = roc_curve(y_test, s)
                auc = roc_auc_score(y_test, s)
            ax.plot(x, yv, label=f"{MODEL_LABELS[name]}  ({auc:.3f})", **MODEL_STYLE[name])
        if kind == "pr":
            ax.scatter([test_metrics["recall"]], [test_metrics["precision"]], s=42, color=INK,
                       edgecolor="white", linewidth=1.5, zorder=6)
            ax.annotate("operating point", (test_metrics["recall"], test_metrics["precision"]),
                        textcoords="offset points", xytext=(8, 6), fontsize=8, color=INK2)
            ax.set(xlabel=f"Recall ({positive})", ylabel=f"Precision ({positive})",
                   title=f"{title} layer: precision-recall on the test period")
            legend_title = "Model  (PR-AUC)"
        else:
            ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8)
            ax.set(xlabel="False positive rate", ylabel="True positive rate",
                   title=f"{title} layer: ROC on the test period")
            legend_title = "Model  (ROC-AUC)"
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(title=legend_title, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  title_fontsize=8, alignment="left")
        _save(fig, fig_dir / f"{kind}_curves")

    # 3. Confusion matrix (row-normalised shading, counts printed).
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#ffffff", "#cde2fb", "#3987e5", "#0d366b"])
    cm = np.array([[test_metrics["tn"], test_metrics["fp"]], [test_metrics["fn"], test_metrics["tp"]]])
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.1%}", ha="center", va="center",
                    fontsize=10, color="white" if share[i, j] > 0.55 else INK)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=[negative, positive],
           yticklabels=[negative, positive], xlabel="Predicted", ylabel="Actual",
           title=f"{title} layer: confusion matrix (test)")
    for s in ax.spines.values():
        s.set_visible(False)
    _save(fig, fig_dir / "confusion_matrix")

    # 4. Performance over time: metrics above, class counts below (separate axes).
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12})
    x = by_step["time_step"]
    a1.plot(x, by_step["f1"], color=INK, lw=2, marker="o", ms=4.5, label="F1")
    a1.plot(x, by_step["precision"], color=BLUE, lw=1.4, marker="o", ms=3, label="Precision")
    a1.plot(x, by_step["recall"], color=ORANGE, lw=1.4, marker="o", ms=3, label="Recall")
    a1.set(ylim=(-0.03, 1.03), ylabel=f"{positive.capitalize()}-class score",
           title=f"{title} layer: test performance per {step_name}")
    a1.legend(ncol=3, loc="lower left")
    a2.bar(x, by_step["n_positive"], color=BLUE, width=0.7)
    a2.set(xlabel=step_name.capitalize(), ylabel=f"# {positive}")
    a2.set_xticks(list(x))
    _save(fig, fig_dir / "per_time_step")

    # 5. Feature importance.
    if importance is not None and len(importance):
        top = importance.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        ax.barh([f[:44] for f in top["feature"]], top["importance"], color=BLUE, height=0.62)
        ax.grid(axis="y", visible=False)
        ax.set(xlabel="Normalised importance (mean over ensemble members)",
               title=f"{title} layer: top 20 features")
        _save(fig, fig_dir / "feature_importance")

    # 6. Score distribution by true class.
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for cls, colour, lab in ((0, BLUE, negative), (1, ORANGE, positive)):
        ax.hist(ensemble_test[y_test == cls], bins=bins, density=True, histtype="stepfilled",
                alpha=0.35, color=colour, edgecolor=colour, linewidth=1.4, label=lab)
    for thr, lab, side in ((review_threshold, "review", "right"), (threshold, "decision", "left")):
        ax.axvline(thr, color=INK2, lw=1)
        ax.text(thr, 1.01, f" {lab} {thr:.2f} ", ha=side, va="bottom", fontsize=8, color=INK2,
                transform=ax.get_xaxis_transform())
    ax.set(yscale="log", xlabel="Ensemble anomaly score", ylabel="Density (log scale)",
           xlim=(-0.02, 1.02))
    ax.set_title(f"{title} layer: score distribution by true class (test)", pad=16)
    ax.legend(loc="upper right")
    _save(fig, fig_dir / "score_distribution")

    # 7. Threshold sweep.
    p, r, t = precision_recall_curve(y_test, ensemble_test)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(t, p[:-1], color=BLUE, lw=1.6, label="Precision")
    ax.plot(t, r[:-1], color=ORANGE, lw=1.6, label="Recall")
    ax.plot(t, f1, color=INK, lw=2.2, label="F1")
    for thr, lab, side in ((review_threshold, "review", "right"), (threshold, "decision", "left")):
        ax.axvline(thr, color=INK2, lw=1)
        ax.text(thr, 1.03, f" {lab} {thr:.2f} ", ha=side, va="bottom", fontsize=8, color=INK2)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Score threshold", ylabel="Score on test period")
    ax.set_title(f"{title} layer: threshold sensitivity", pad=16)
    ax.legend(loc="lower center", ncol=3)
    _save(fig, fig_dir / "threshold_sweep")

    # 8. Reliability diagram.
    frac, mean_pred = calibration_curve(y_test, np.clip(ensemble_test, 0, 1), n_bins=10,
                                        strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8, label="Perfect calibration")
    ax.plot(mean_pred, frac, color=INK, lw=2, marker="o", ms=5, label="Ensemble")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Mean predicted score",
           ylabel=f"Observed {positive} fraction",
           title=f"{title} layer: calibration (test)")
    ax.legend(loc="upper left")
    _save(fig, fig_dir / "calibration")


# --------------------------------------------------------------------------- #
# The shared experiment
# --------------------------------------------------------------------------- #
def check_split(args, steps: np.ndarray) -> None:
    if not (1 <= args.val_start <= args.train_end < steps.max()):
        raise ValueError("Need 1 <= --val-start <= --train-end < last time step")


def fit_evaluate_report(X, names, y_raw, steps, ids, args, *, id_col: str, entity: str, t0: float,
                        n_base: int | None = None, model_meta: dict | None = None,
                        rf_seeds: int | None = None,
                        verdict_names=("licit", "suspicious", "illicit"),
                        step_name: str = "time step", extra_cols: dict | None = None):
    """Run the full protocol and write every output. Returns (results dict, scores frame).

    X        (n, d) float32, original features in the first `n_base` columns
    y_raw    1 positive, 0 negative, -1 unlabelled
    steps    integer time step used for the temporal split
    Baselines see only the first `n_base` columns (the features prior work used).
    """
    negative, _, positive = verdict_names
    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    check_split(args, steps)
    n_base = n_base or X.shape[1]
    log.info("Feature matrix: %s (%d original features)", X.shape, n_base)
    in_train = steps <= args.train_end
    labelled = y_raw != UNKNOWN
    masks = {"fit": labelled & (steps < args.val_start),
             "validation": labelled & (steps >= args.val_start) & in_train,
             "train (fit+val)": labelled & in_train,
             "test": labelled & ~in_train}
    fit_m, val_m, train_m, test_m = masks.values()
    y = y_raw.astype(np.int32)

    split_rows = []
    for label, m in masks.items():
        if m.sum() == 0 or y[m].sum() in (0, m.sum()):
            raise ValueError(f"The {label} split needs both classes; adjust the split arguments")
        period = (steps >= steps[m].min()) & (steps <= steps[m].max())
        split_rows.append({"split": label, f"{step_name}s": f"{steps[m].min()}-{steps[m].max()}",
                           "labelled": int(m.sum()), positive: int(y[m].sum()),
                           negative: int(m.sum() - y[m].sum()),
                           f"{positive} %": 100 * float(y[m].mean()),
                           "unlabelled": int((period & ~labelled).sum())})
        log.info("%-16s steps %s: %7d labelled, %6d %s (%.1f%%)", label,
                 split_rows[-1][f"{step_name}s"], m.sum(), y[m].sum(), positive, 100 * y[m].mean())
    split_table = pd.DataFrame(split_rows)

    members = available_models(args.models)
    baselines = [b for b in args.baselines if b in BASELINES]
    seeds = [args.seed + i for i in range(args.n_seeds)]
    model_seeds = {m: seeds[:rf_seeds] if (m == "rf" and rf_seeds) else seeds for m in members}
    model_seeds.update({b: seeds[:1] for b in baselines})
    cols = {m: slice(None) for m in members} | {b: slice(0, n_base) for b in baselines}
    X_fit, X_val = X[fit_m], X[val_m]

    # Phase 1: fit on early steps; early stopping and thresholds from validation.
    val_seed_scores, rounds = {}, {}
    for name in baselines + members:
        preds, used = [], []
        for seed in model_seeds[name]:
            model, n = TRAINERS[name](X_fit[:, cols[name]], y[fit_m], X_val[:, cols[name]],
                                      y[val_m], seed)
            preds.append(proba(model, X_val[:, cols[name]], n))
            used.append(n)
            del model
        val_seed_scores[name], rounds[name] = preds, used
        log.info("[val] %-10s PR-AUC=%.4f rounds=%s", name,
                 average_precision_score(y[val_m], np.mean(preds, axis=0)), used)
    val_scores = {k: np.mean(v, axis=0) for k, v in val_seed_scores.items()}
    val_ensemble = np.mean([val_scores[m] for m in members], axis=0)
    threshold = best_f1_threshold(y[val_m], val_ensemble)
    review_threshold = min(threshold, recall_threshold(y[val_m], val_ensemble, args.review_recall))
    thresholds = {k: FIXED_THRESHOLD.get(k, best_f1_threshold(y[val_m], v))
                  for k, v in val_scores.items()}
    val_metrics = evaluate(y[val_m], val_ensemble, threshold)
    log.info("[val] ensemble   PR-AUC=%.4f F1=%.4f | decision threshold %.4f, review threshold %.4f",
             val_metrics["pr_auc"], val_metrics["f1"], threshold, review_threshold)
    del X_fit, X_val

    # Phase 2: refit on all training steps; score every entity once.
    grow = min(train_m.sum() / fit_m.sum(), 1.25)   # more data -> proportionally more rounds
    X_train = X[train_m]
    models, all_scores, seed_f1 = {}, {}, {}
    for name in baselines + members:
        fitted, preds, f1s = [], [], []
        for seed, n, val_s in zip(model_seeds[name], rounds[name], val_seed_scores[name]):
            n_final = max(1, int(round(n * grow))) if n else None
            model, _ = TRAINERS[name](X_train[:, cols[name]], y[train_m], None, None, seed, n_final)
            s = proba(model, X[:, cols[name]])
            thr = FIXED_THRESHOLD.get(name, best_f1_threshold(y[val_m], val_s))
            f1s.append(evaluate(y[test_m], s[test_m], thr)["f1"])
            preds.append(s)
            if name in members:
                fitted.append(model)
        if name in members:
            models[name] = fitted
        all_scores[name], seed_f1[name] = np.mean(preds, axis=0), f1s
        log.info("[refit] %-10s done", name)
    del X_train
    ensemble = np.mean([all_scores[m] for m in members], axis=0)

    # Test-period evaluation.
    y_test = y[test_m]
    test_metrics = evaluate(y_test, ensemble[test_m], threshold)
    ci = bootstrap_ci(y_test, ensemble[test_m], threshold, args.bootstrap, args.seed)
    per_model = {k: evaluate(y_test, s[test_m], thresholds[k]) for k, s in all_scores.items()}
    by_step = per_time_step(y_test, ensemble[test_m], steps[test_m], threshold)
    importance = feature_importance(models, names)

    def row(name, m):
        kind = "ensemble" if name == "ensemble" else "baseline" if name in baselines else "member"
        std = float(np.std(seed_f1[name])) if len(seed_f1.get(name, [])) > 1 else float("nan")
        return {"Model": MODEL_LABELS[name], "Role": kind,
                "Features": "original" if name in baselines and n_base < X.shape[1] else "all",
                "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
                "F1 seed std": std, "FPR": m["fpr"], "MCC": m["mcc"],
                "PR-AUC": m["pr_auc"], "ROC-AUC": m["roc_auc"], "Micro-F1": m["micro_f1"]}
    model_table = pd.DataFrame([row(k, v) for k, v in per_model.items()]
                               + [row("ensemble", test_metrics)])

    verdict_idx = (ensemble >= review_threshold).astype(int) + (ensemble >= threshold).astype(int)
    verdict = np.array(verdict_names, dtype=object)[verdict_idx]
    truth = pd.Series(y_raw).map({ILLICIT: positive, LICIT: negative, UNKNOWN: "unlabelled"})
    scores = pd.DataFrame({
        id_col: ids, step_name.replace(" ", "_"): steps, **(extra_cols or {}),
        "split": np.where(in_train, "train", "test"), "true_label": truth.to_numpy(),
        "anomaly_score": ensemble, "verdict": verdict,
        **{f"score_{k}": all_scores[k] for k in members},
    })
    test_all = ~in_train
    verdict_table = (pd.crosstab(scores.loc[test_all, "true_label"], scores.loc[test_all, "verdict"])
                     .reindex(index=[positive, negative, "unlabelled"],
                              columns=list(verdict_names), fill_value=0)
                     .rename_axis("True label (test period)").reset_index())
    verdict_table["total"] = verdict_table[list(verdict_names)].sum(axis=1)

    ci_table = pd.DataFrame([
        {"Metric": k, "Estimate": test_metrics[k], "CI 2.5%": lo, "CI 97.5%": hi}
        for k, (lo, hi) in ci.items()])
    cm_table = pd.DataFrame({"": [f"actual {negative}", f"actual {positive}"],
                             f"predicted {negative}": [test_metrics["tn"], test_metrics["fn"]],
                             f"predicted {positive}": [test_metrics["fp"], test_metrics["tp"]]})

    tdir, name = out_dir / "tables", entity
    write_table(split_table, tdir / "dataset_split", f"{name.capitalize()} layer: temporal data split.",
                f"tab:{name}-split", digits=1)
    write_table(model_table, tdir / "model_comparison",
                f"{name.capitalize()} layer: {positive}-class results on the test period. "
                "Thresholds are chosen on validation data only.", f"tab:{name}-models")
    write_table(ci_table, tdir / "bootstrap_ci",
                f"{name.capitalize()} layer: ensemble test metrics with 95% bootstrap intervals "
                f"({args.bootstrap} resamples).", f"tab:{name}-ci")
    write_table(cm_table, tdir / "confusion_matrix",
                f"{name.capitalize()} layer: ensemble confusion matrix on the test period.",
                f"tab:{name}-cm")
    write_table(by_step, tdir / "per_time_step",
                f"{name.capitalize()} layer: ensemble test results per {step_name}.",
                f"tab:{name}-steps")
    write_table(verdict_table, tdir / "verdict_summary",
                f"{name.capitalize()} layer: three-way verdicts on the test period.",
                f"tab:{name}-verdicts")
    if importance is not None:
        write_table(importance.head(30), tdir / "feature_importance",
                    f"{name.capitalize()} layer: top 30 features.", f"tab:{name}-features", digits=4)

    results = {
        "entity": entity, "positive_class": positive, "step_name": step_name,
        "split": {"fit_steps": [int(steps[fit_m].min()), args.val_start - 1],
                  "val_steps": [args.val_start, args.train_end],
                  "test_steps": [args.train_end + 1, int(steps.max())]},
        "n_entities": int(len(y_raw)), "n_features": int(X.shape[1]), "n_original_features": int(n_base),
        "graph_features": bool((model_meta or {}).get("graph_features", False)),
        "models": members, "baselines": baselines, "seeds": model_seeds, "boosting_rounds": rounds,
        "decision_threshold": threshold, "review_threshold": review_threshold,
        "validation": val_metrics, "test": test_metrics, "test_ci95": ci,
        "test_per_model": per_model, "test_f1_per_seed": seed_f1,
        "dataset_split": split_rows,
        "verdicts_test": verdict_table.to_dict(orient="records"),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    scores.to_csv(out_dir / f"{entity}_verdicts.csv", index=False)
    joblib.dump({"models": models, "decision_threshold": threshold,
                 "review_threshold": review_threshold, "feature_names": names,
                 "verdict_names": verdict_names, **(model_meta or {})},
                out_dir / "model.joblib", compress=3)
    make_figures(out_dir / "figures", y_test=y_test,
                 test_scores={k: s[test_m] for k, s in all_scores.items()},
                 ensemble_test=ensemble[test_m], threshold=threshold,
                 review_threshold=review_threshold, by_step=by_step, importance=importance,
                 test_metrics=test_metrics, entity=entity, positive=positive, negative=negative,
                 step_name=step_name)

    print("\n" + "=" * 78)
    print(f"{entity.upper()} LAYER - TEST ({step_name}s {args.train_end + 1}-{steps.max()}), "
          f"{positive} class")
    print("=" * 78)
    print(f"{'model':<36}{'prec':>7}{'recall':>8}{'F1':>7}{'PR-AUC':>8}{'ROC-AUC':>9}{'MCC':>7}")
    for r in model_table.itertuples(index=False):
        print(f"{r.Model:<36}{r.Precision:>7.3f}{r.Recall:>8.3f}{r.F1:>7.3f}"
              f"{r[9]:>8.3f}{r[10]:>9.3f}{r.MCC:>7.3f}")
    if "f1" in ci:
        print(f"\nEnsemble F1 95% CI [{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}]")
    m = test_metrics
    print(f"Decision threshold {threshold:.4f}, review threshold {review_threshold:.4f} | "
          f"TP {m['tp']}  FP {m['fp']}  FN {m['fn']}  TN {m['tn']}")
    print("Verdicts, test period:")
    print(verdict_table.to_string(index=False))
    print(f"Outputs written to {out_dir.resolve()}  ({results['runtime_seconds']}s)")
    return results, scores


# --------------------------------------------------------------------------- #
# CLI plumbing
# --------------------------------------------------------------------------- #
def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", default="Dataset_elliptic++",
                   help="folder with the Elliptic++ csv files")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--train-end", type=int, default=34, help="last training time step")
    p.add_argument("--val-start", type=int, default=30,
                   help="first validation time step (validation = val-start..train-end)")
    p.add_argument("--models", nargs="+", default=["lgbm", "xgb", "rf"],
                   choices=["lgbm", "xgb", "rf", "hgb"], help="ensemble members")
    p.add_argument("--baselines", nargs="*", default=list(BASELINES), choices=list(BASELINES),
                   help="reference models reported next to the ensemble (pass none to skip)")
    p.add_argument("--n-seeds", type=int, default=3, help="seeds averaged per ensemble member")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--review-recall", type=float, default=0.95,
                   help="validation recall kept by the 'suspicious' review threshold")
    p.add_argument("--bootstrap", type=int, default=500,
                   help="bootstrap resamples for 95%% confidence intervals (0 = off)")
    p.add_argument("--no-graph-features", action="store_true",
                   help="ablation: use only the original tabular features")
    p.add_argument("--no-cache", action="store_true", help="do not read/write cache files")
    return p


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for noisy in ("fontTools", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    warnings.filterwarnings("ignore", message=".*'eval_set' is deprecated")
