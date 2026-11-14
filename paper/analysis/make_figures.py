"""Figures for the manuscript (vector PDF for LaTeX, SVG for the HTML build).

Sized for IEEE two-column pages: 3.5 in (one column) or 7.16 in (two columns),
8 pt Times-like type. Reads only saved outputs and paper/analysis/results/.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import average_precision_score, precision_recall_curve

from common import RESULTS, ROOT

FIG = ROOT / "paper" / "manuscript" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COL, WIDE = 3.5, 7.16
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, LIGHTBLUE = "#2a78d6", "#eb6834", "#1baf7a", "#9ec5f4"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.2,
    "figure.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "svg.fonttype": "path",
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.svg")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"[figure] {name}", flush=True)


def have(*names):
    missing = [n for n in names if not (RESULTS / n).exists()]
    if missing:
        print(f"[skip] missing {missing}")
    return not missing


# --------------------------------------------------------------------- Fig. overview
def fig_overview():
    import fig_overview as ov
    save(ov.draw(WIDE), "fig_overview")


# ------------------------------------------------------------- Fig. drift (transactions)
def fig_tx_drift():
    if not have("tx_prequential_per_step.csv"):
        return
    d = pd.read_csv(RESULTS / "tx_prequential_per_step.csv")
    st, rt = d[d.model == "static"], d[d.model == "retrained"]
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(COL, 3.7), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 2, 1], "hspace": 0.16})
    for ax, col, lab in ((a0, "pr_auc", "PR-AUC within step"), (a1, "f1", "F1 at threshold 0.5")):
        ax.axvspan(42.5, 49.5, color="#f4f3ef", lw=0)
        ax.plot(st.step, st[col], color=INK, marker="o", ms=3, label="Static (trained on steps 1–34)")
        ax.plot(rt.step, rt[col], color=BLUE, marker="s", ms=3, label="Retrained before every step")
        ax.set(ylim=(-0.05, 1.05), ylabel=lab)
    a1.text(46, 0.98, "after dark-market\nshutdown", ha="center", va="top", fontsize=6.5, color=INK2)
    a0.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, columnspacing=1.2,
              handlelength=1.6)
    a2.axvspan(42.5, 49.5, color="#f4f3ef", lw=0)
    a2.bar(st.step, st.n_pos, color=LIGHTBLUE, width=0.7)
    for s_, n_ in zip(st.step, st.n_pos):
        if n_ < 30:
            a2.text(s_, n_ + 8, f"{int(n_)}", ha="center", va="bottom", fontsize=5.8, color=INK2)
    a2.set(xlabel="Time step", ylabel="# illicit")
    a2.set_xticks(range(35, 50))
    save(fig, "fig_tx_drift")


# ------------------------------------------------------------------ Fig. wallet ablation
def fig_wallet_ablation():
    if not have("wallet_feature_ablation.csv"):
        return
    d = pd.read_csv(RESULTS / "wallet_feature_ablation.csv")
    order = ["original (55)", "+ structural (9)", "+ in-neighbour means", "+ out-neighbour means",
             "+ both neighbour means", "all (174)", "graph only (119)"]
    labels = ["Original features (55)", "+ structural (9)", "+ in-neighbour means (55)",
              "+ out-neighbour means (55)", "+ both neighbour means", "All features (174)",
              "Graph-derived only (119)"]
    g = d.groupby("features")
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    y = np.arange(len(order))[::-1]
    for off, col, colr, lab in ((0.19, "f1", INK, "F1"), (-0.19, "pr_auc", BLUE, "PR-AUC")):
        mean = g[col].mean().reindex(order).to_numpy()
        lo, hi = g[col].min().reindex(order).to_numpy(), g[col].max().reindex(order).to_numpy()
        ax.barh(y + off, mean, height=0.34, color=colr, label=lab)
        ax.errorbar(mean, y + off, xerr=[mean - lo, hi - mean], fmt="none", ecolor=MUTED,
                    elinewidth=0.9, capsize=1.6)
        for yy, m in zip(y + off, hi):
            ax.text(m + 0.012, yy, f"{mean[list(y + off).index(yy)]:.2f}", va="center", fontsize=6.3,
                    color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set(xlim=(0, 0.92), xlabel="Illicit-class score on test steps 35–49 (mean and range of 3 seeds)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2)
    save(fig, "fig_wallet_ablation")


# --------------------------------------------------------------------- Fig. PR curves
def fig_pr_curves():
    panels = [("transaction", "outputs/transaction", "illicit", "(a) Transactions"),
              ("wallet", "outputs/wallet", "illicit", "(b) Wallets"),
              ("network", "paper/analysis/results/network_full", "attack", "(c) Network windows")]
    if not (ROOT / panels[2][1] / "metrics.json").exists():
        panels[2] = ("network", "outputs/network", "attack", "(c) Network windows (sampled)")
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.25), sharey=True)
    style = {"score_lgbm": (BLUE, "LightGBM", 1.0), "score_xgb": (ORANGE, "XGBoost", 1.0),
             "score_rf": (AQUA, "RF", 1.0), "anomaly_score": (INK, "Ensemble", 1.7)}
    for ax, (entity, folder, pos, title) in zip(axes, panels):
        meta = json.loads((ROOT / folder / "metrics.json").read_text())
        df = pd.read_csv(ROOT / folder / f"{entity}_verdicts.csv", low_memory=False)
        t = df[(df.split == "test") & (df.true_label != "unlabelled")]
        y = (t.true_label == pos).to_numpy()
        for col, (colr, lab, lw) in style.items():
            p, r, _ = precision_recall_curve(y, t[col].to_numpy())
            ax.plot(r, p, color=colr, lw=lw, label=f"{lab} ({average_precision_score(y, t[col]):.3f})",
                    zorder=3 if col == "anomaly_score" else 2)
        m = meta["test"]
        ax.scatter([m["recall"]], [m["precision"]], s=16, color=INK, edgecolor="white", linewidth=0.8, zorder=5)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", title=title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, title="Model (PR-AUC)",
                  title_fontsize=6.5, fontsize=6.2, handlelength=1.2, columnspacing=0.8)
    axes[0].set_ylabel("Precision")
    save(fig, "fig_pr_curves")


# -------------------------------------------------------------- Fig. random vs temporal
def fig_split_protocol():
    if not have("tx_random_vs_temporal.csv", "wallet_random_vs_temporal.csv"):
        return
    tx = pd.read_csv(RESULTS / "tx_random_vs_temporal.csv")
    wa = pd.read_csv(RESULTS / "wallet_random_vs_temporal.csv")
    rows = []
    for model in ("RF-50 (original features)", "LightGBM (all features)"):
        for split in ("temporal", "random"):
            rows.append((f"Tx: {model.split(' (')[0]}", split, tx[(tx.model == model) & (tx.split == split)].f1))
    w1 = wa[wa.rows == "one per wallet"]
    for model in ("RF-50 (original features)", "LightGBM (all features)"):
        rows.append((f"Wallet: {model.split(' (')[0]}", "temporal",
                     w1[(w1.model == model) & w1.split.str.startswith("temporal")].f1))
        rows.append((f"Wallet: {model.split(' (')[0]}", "random",
                     w1[(w1.model == model) & w1.split.str.startswith("random")].f1))
    w2 = wa[wa.rows != "one per wallet"]
    rows.append(("Wallet rows as\ndistributed: RF-50", "temporal", w2[w2.split.str.startswith("temporal")].f1))
    rows.append(("Wallet rows as\ndistributed: RF-50", "random", w2[w2.split.str.startswith("random")].f1))
    groups = list(dict.fromkeys(r[0] for r in rows))
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    y = np.arange(len(groups))[::-1]
    for off, split, colr, lab in ((0.19, "temporal", INK, "Temporal split"), (-0.19, "random", ORANGE, "Random 70/30 split")):
        vals = [next(r[2] for r in rows if r[0] == g and r[1] == split) for g in groups]
        mean = np.array([v.mean() for v in vals])
        ax.barh(y + off, mean, height=0.34, color=colr, label=lab)
        for yy, m in zip(y + off, mean):
            ax.text(m + 0.012, yy, f"{m:.2f}", va="center", fontsize=6.3, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.set(xlim=(0, 1.1), xlabel="Illicit F1 at threshold 0.5")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.14), ncol=2)
    save(fig, "fig_split_protocol")


# ----------------------------------------------------------- Fig. wallet drift per step
def fig_wallet_drift():
    if not have("wallet_prequential_per_step.csv"):
        return
    d = pd.read_csv(RESULTS / "wallet_prequential_per_step.csv")
    st, rt = d[d.model == "static"], d[d.model == "retrained"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 2.75), sharex=True,
                                 gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.14})
    a1.plot(st.step, st.pr_auc, color=INK, marker="o", ms=3, label="Static (trained on steps 1–34)")
    a1.plot(rt.step, rt.pr_auc, color=BLUE, marker="s", ms=3, label="Retrained before every step")
    a1.set(ylim=(-0.04, 1.04), ylabel="Illicit PR-AUC within step")
    a1.legend(loc="lower left")
    a2.bar(st.step, st.n_pos, color=LIGHTBLUE, width=0.7)
    a2.set(xlabel="Time step of first appearance", ylabel="# illicit")
    a2.set_xticks(range(35, 50))
    save(fig, "fig_wallet_drift")


# -------------------------------------------------- Fig. score distributions by period
def fig_tx_scores():
    df = pd.read_csv(ROOT / "outputs" / "transaction" / "transaction_verdicts.csv")
    meta = json.loads((ROOT / "outputs" / "transaction" / "metrics.json").read_text())
    t = df[(df.split == "test") & (df.true_label != "unlabelled")]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.75), sharey=True)
    bins = np.linspace(0, 1, 26)
    for ax, (lo, hi, title) in zip(axes, ((35, 42, "Steps 35–42"), (43, 49, "Steps 43–49"))):
        p = t[(t.time_step >= lo) & (t.time_step <= hi)]
        for lab, colr in (("licit", LIGHTBLUE), ("illicit", ORANGE)):
            s = p[p.true_label == lab].anomaly_score
            ax.hist(s, bins=bins, weights=np.full(len(s), 1 / max(len(s), 1)), histtype="stepfilled",
                    alpha=0.55, color=colr, edgecolor=colr, lw=0.8, label=f"{lab} (n={len(s):,})")
        ax.axvline(meta["decision_threshold"], color=INK, lw=0.7)
        ax.set(title=title, xlabel="Ensemble score", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc="upper center", fontsize=6.2)
    axes[0].set_ylabel("Share of class")
    save(fig, "fig_tx_scores")


if __name__ == "__main__":
    fig_overview()
    fig_tx_scores()
    fig_pr_curves()
    fig_tx_drift()
    fig_wallet_ablation()
    fig_split_protocol()
    fig_wallet_drift()
