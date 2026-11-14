"""
Run every experiment and assemble the research report.

Experiments
-----------
    transaction            Layer 1, full model (tabular + graph features)
    transaction_no_graph   Layer 1 ablation: tabular features only
    wallet                 Layer 2, full model, leak-free first-seen protocol
    wallet_no_graph        Layer 2 ablation: tabular features only
    wallet_paper_protocol  Layer 2 under the row-per-appearance protocol
    network                Layer 3, UGR'16 (skipped until a flow file is present)

Each experiment writes to outputs/<name>/. Finished experiments are skipped on
re-runs (use --force to redo them). The report is rebuilt every time:

    report/REPORT.md        results section with every table, figures linked
    report/tables.tex       all LaTeX tables in one file
    report/figures/         publication figures (PNG 300 dpi + PDF), per layer

Usage
-----
    python run_all.py                    # run what is missing, then build the report
    python run_all.py --report-only      # just rebuild the report
    python run_all.py --only network     # run a single experiment
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from anomaly_core import MODEL_LABELS, to_latex, to_markdown

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    "transaction": ["txn_anomaly_detection.py"],
    "transaction_no_graph": ["txn_anomaly_detection.py", "--no-graph-features"],
    "wallet": ["wallet_anomaly_detection.py"],
    "wallet_no_graph": ["wallet_anomaly_detection.py", "--no-graph-features"],
    "wallet_paper_protocol": ["wallet_anomaly_detection.py", "--protocol", "paper"],
    "network": ["network_anomaly_detection.py"],
}

# Illicit-class results as printed in Elmougy & Liu, "Demystifying Fraudulent Transactions
# and Illicit Nodes in the Bitcoin Network for Financial Forensics", KDD 2023
# (arXiv:2306.06108v1, Tables 7 and 8). Split: time steps 1-34 train, 35-49 test.
PUBLISHED = {
    "transaction": [("Logistic Regression", 0.328, 0.707, 0.448), ("Random Forest", 0.975, 0.719, 0.828),
                    ("MLP", 0.611, 0.613, 0.612), ("LSTM", 0.709, 0.223, 0.339),
                    ("XGBoost", 0.793, 0.718, 0.754), ("RF+MLP+XGB ensemble", 0.962, 0.723, 0.826)],
    "wallet": [("Logistic Regression", 0.477, 0.046, 0.083), ("Random Forest", 0.911, 0.789, 0.845),
               ("MLP", 0.708, 0.502, 0.587), ("LSTM", 0.922, 0.033, 0.064),
               ("XGBoost", 0.869, 0.534, 0.662), ("RF+MLP+XGB ensemble", 0.933, 0.572, 0.709)],
}
LAYER_TITLES = {"transaction": "Layer 1: transactions (Elliptic++)",
                "wallet": "Layer 2: wallets (Elliptic++)",
                "network": "Layer 3: network traffic (UGR'16)"}
FIGURES = [("pr_curves", "Precision-recall curves, all models"),
           ("roc_curves", "ROC curves, all models"),
           ("confusion_matrix", "Confusion matrix of the ensemble"),
           ("per_time_step", "Performance over time"),
           ("threshold_sweep", "Threshold sensitivity"),
           ("score_distribution", "Score distribution by true class"),
           ("calibration", "Reliability diagram"),
           ("feature_importance", "Top features")]


def load(out_root: Path, name: str) -> dict | None:
    f = out_root / name / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else None


def run_experiments(args, out_root: Path) -> None:
    for name, cmd in EXPERIMENTS.items():
        if args.only and name not in args.only:
            continue
        out_dir = out_root / name
        if (out_dir / "metrics.json").exists() and not args.force:
            print(f"[skip] {name}: already done ({out_dir})")
            continue
        extra = ["--out-dir", str(out_dir)]
        if name == "network":
            src = Path(args.ugr_input)
            if not src.exists() or (src.is_dir() and not any(src.iterdir())):
                print(f"[skip] network: no UGR'16 file at {src} yet")
                continue
            extra += ["--input", str(src)]
            if args.max_flows:
                extra += ["--max-flows", str(args.max_flows)]
            if args.background_rate is not None:
                extra += ["--background-rate", str(args.background_rate)]
        else:
            extra += ["--data-dir", args.data_dir]
        print(f"\n[run] {name}", flush=True)
        done = subprocess.run([sys.executable, str(ROOT / cmd[0]), *cmd[1:], *extra], cwd=ROOT)
        if done.returncode != 0:
            print(f"[failed] {name} exited with code {done.returncode}")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def ci_text(res: dict, key: str) -> str:
    lo, hi = res.get("test_ci95", {}).get(key, (None, None))
    return f"{res['test'][key]:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None else f"{res['test'][key]:.3f}"


def headline_table(runs: dict) -> pd.DataFrame:
    rows = []
    for name in ("transaction", "wallet", "network"):
        r = runs.get(name)
        if r:
            t = r["dataset_split"][-1]
            rows.append({"Layer": LAYER_TITLES[name], "Test set": f"{t['labelled']:,} labelled",
                         "Positives": f"{r['test']['tp'] + r['test']['fn']:,} {r['positive_class']}",
                         "Precision [95% CI]": ci_text(r, "precision"),
                         "Recall [95% CI]": ci_text(r, "recall"), "F1 [95% CI]": ci_text(r, "f1"),
                         "PR-AUC": r["test"]["pr_auc"], "ROC-AUC": r["test"]["roc_auc"],
                         "MCC": r["test"]["mcc"]})
    return pd.DataFrame(rows)


def literature_table(runs: dict, layer: str) -> pd.DataFrame:
    rows = [{"Source": "Elmougy & Liu (2023), as published", "Model": m, "Precision": p,
             "Recall": r, "F1": f} for m, p, r, f in PUBLISHED[layer]]
    res = runs[layer]
    repro = res["test_per_model"].get("rf_default")
    if repro:
        rows.append({"Source": "Our reproduction (same split, original features, threshold 0.5)",
                     "Model": "Random Forest (50 trees, default)", "Precision": repro["precision"],
                     "Recall": repro["recall"], "F1": repro["f1"]})
    for key in res["models"]:
        m = res["test_per_model"][key]
        rows.append({"Source": "This work", "Model": MODEL_LABELS[key], "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"]})
    rows.append({"Source": "This work", "Model": "Ensemble (ours)", "Precision": res["test"]["precision"],
                 "Recall": res["test"]["recall"], "F1": res["test"]["f1"]})
    return pd.DataFrame(rows)


def ablation_table(runs: dict) -> pd.DataFrame:
    rows = []
    for layer in ("transaction", "wallet"):
        full, plain = runs.get(layer), runs.get(f"{layer}_no_graph")
        if not (full and plain):
            continue
        for label, r in (("Original features only", plain), ("+ graph features (full model)", full)):
            rows.append({"Layer": layer, "Features": label, "# features": r["n_features"],
                         "Precision": r["test"]["precision"], "Recall": r["test"]["recall"],
                         "F1": r["test"]["f1"], "PR-AUC": r["test"]["pr_auc"],
                         "ROC-AUC": r["test"]["roc_auc"]})
        rows.append({"Layer": layer, "Features": "Change from graph features", "# features":
                     full["n_features"] - plain["n_features"],
                     **{k2: full["test"][k] - plain["test"][k] for k, k2 in
                        (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
                         ("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"))}})
    return pd.DataFrame(rows)


def drift_table(out_root: Path) -> pd.DataFrame | None:
    """Transaction layer before / after the dark-market shutdown (time step 43)."""
    f = out_root / "transaction" / "tables" / "per_time_step.csv"
    if not f.exists():
        return None
    steps = pd.read_csv(f)
    rows = []
    for label, part in (("Steps 35-42 (before shutdown)", steps[steps.time_step <= 42]),
                        ("Steps 43-49 (after shutdown)", steps[steps.time_step >= 43]),
                        ("All test steps", steps)):
        tp, fp, fn = part.tp.sum(), part.fp.sum(), part.fn.sum()
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        rows.append({"Period": label, "Illicit": int(tp + fn), "Detected": int(tp),
                     "False alarms": int(fp), "Precision": p, "Recall": r,
                     "F1": 2 * p * r / max(p + r, 1e-12)})
    return pd.DataFrame(rows)


def build_report(out_root: Path, report_dir: Path) -> None:
    runs = {name: load(out_root, name) for name in EXPERIMENTS}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        print("No finished experiments yet - nothing to report.")
        return
    report_dir.mkdir(parents=True, exist_ok=True)
    md, tex = [], []

    def add(title: str, df: pd.DataFrame | None, label: str, note: str = "", digits: int = 3):
        if df is None or df.empty:
            return
        md.append(f"**Table: {title}**\n\n{to_markdown(df, digits)}\n")
        if note:
            md.append(note + "\n")
        tex.append(to_latex(df, title, f"tab:{label}", digits))

    def add_file(layer: str, table: str, title: str, label: str, digits: int = 3):
        f = out_root / layer / "tables" / f"{table}.csv"
        if f.exists():
            add(title, pd.read_csv(f), label, digits=digits)

    md.append("# Multi-layer anomaly detection: results\n")
    md.append("Generated by `run_all.py` from the files in `outputs/`. Every number below is "
              "measured on a test period that was never used for training, early stopping or "
              "threshold selection. Bracketed ranges are 95% bootstrap intervals over the test set.\n")
    missing = [n for n in ("transaction", "wallet", "network") if n not in runs]
    if missing:
        md.append(f"> Not yet run: {', '.join(missing)}. Those sections are absent below.\n")

    md.append("## 1. Headline results\n")
    add("Ensemble performance on the positive class, per layer.", headline_table(runs), "headline")

    md.append("## 2. Experimental protocol\n")
    md.append(
        "- **Temporal split.** Models are fitted on the earliest time steps, tuned on the last "
        "training steps (validation) and evaluated on later, unseen steps. No random shuffling.\n"
        "- **Model.** Average of LightGBM, XGBoost and Random Forest, several seeds each. Boosting "
        "rounds come from early stopping on validation; models are then refitted on fit + validation.\n"
        "- **Thresholds.** The decision threshold maximises validation F1. A lower review threshold "
        "keeps 95% validation recall. Scores between the two give the verdict *suspicious*.\n"
        "- **Baselines.** Logistic regression, an unsupervised Isolation Forest fitted on the "
        "negative class only, and the default 50-tree Random Forest used in prior work. Baselines "
        "use the original features only.\n"
        "- **Graph features (layers 1-2).** Degrees, 2-hop reach, component size, and the mean "
        "feature vector of in- and out-neighbours. Features are propagated, labels never are.\n")
    for layer in ("transaction", "wallet", "network"):
        if layer in runs:
            add_file(layer, "dataset_split", f"{LAYER_TITLES[layer]} - temporal data split.",
                     f"{layer}-split", digits=1)

    md.append("## 3. Model comparison per layer\n")
    for layer in ("transaction", "wallet", "network"):
        if layer in runs:
            add_file(layer, "model_comparison",
                     f"{LAYER_TITLES[layer]} - all models on the test period.", f"{layer}-models")
            add_file(layer, "bootstrap_ci",
                     f"{LAYER_TITLES[layer]} - ensemble metrics with 95% bootstrap intervals.",
                     f"{layer}-ci")

    md.append("## 4. Comparison with published results\n")
    md.append("Published numbers are copied from Elmougy & Liu (KDD 2023, arXiv:2306.06108v1, "
              "Tables 7 and 8), who use the same split (steps 1-34 train, 35-49 test). The "
              "*reproduction* row is their strongest single model re-run inside this pipeline.\n")
    for layer in ("transaction", "wallet"):
        if layer in runs:
            add(f"{LAYER_TITLES[layer]} - illicit-class results against prior work.",
                literature_table(runs, layer), f"{layer}-literature")
    if "wallet" in runs and "rf_default" in runs["wallet"]["test_per_model"]:
        pub = dict((m, f) for m, _, _, f in PUBLISHED["wallet"])["Random Forest"]
        got = runs["wallet"]["test_per_model"]["rf_default"]["f1"]
        md.append(f"The published wallet Random Forest F1 ({pub:.3f}) is not reproduced here "
                  f"({got:.3f} with the same model settings on a time-based split). The published "
                  "paper does not state how wallets that appear in several time steps are assigned "
                  "to train or test, so the two numbers may not measure the same task; treat the "
                  "comparison as indicative.\n")
    if "wallet_paper_protocol" in runs and "wallet" in runs:
        a, b = runs["wallet"]["test"], runs["wallet_paper_protocol"]["test"]
        add("Wallet layer - effect of the evaluation protocol.", pd.DataFrame([
            {"Protocol": "first-seen (one row per wallet, leak-free)", "Precision": a["precision"],
             "Recall": a["recall"], "F1": a["f1"], "PR-AUC": a["pr_auc"]},
            {"Protocol": "row per wallet per time step", "Precision": b["precision"],
             "Recall": b["recall"], "F1": b["f1"], "PR-AUC": b["pr_auc"]}]), "wallet-protocol")

    md.append("## 5. Ablation: contribution of graph features\n")
    add("Ensemble with and without graph-derived features.", ablation_table(runs), "ablation")

    md.append("## 6. Robustness over time\n")
    drift = drift_table(out_root)
    note = ""
    if drift is not None and drift["F1"][1] < drift["F1"][0] - 0.2:
        note = (f"The model is trained on steps 1-34. Pooled F1 falls from {drift['F1'][0]:.3f} "
                f"before the shutdown to {drift['F1'][1]:.3f} after it: the illicit transactions "
                "that remain behave differently from anything seen in training. This concept "
                "drift is the main limitation of a model that is trained once and not updated.")
    add("Transaction layer - pooled results before and after the dark-market shutdown "
        "(time step 43).", drift, "drift", note=note)
    for layer in ("wallet", "network"):
        if layer in runs:
            add_file(layer, "per_time_step",
                     f"{LAYER_TITLES[layer]} - ensemble results per {runs[layer]['step_name']}.",
                     f"{layer}-steps")

    md.append("## 7. Final verdicts\n")
    md.append("Every entity receives one of three verdicts. The full lists are in "
              "`outputs/<layer>/<layer>_verdicts.csv`.\n")
    for layer in ("transaction", "wallet", "network"):
        if layer in runs:
            add_file(layer, "verdict_summary",
                     f"{LAYER_TITLES[layer]} - verdicts on the test period by true label.",
                     f"{layer}-verdicts")

    if "network" in runs:
        md.append("## 8. Network layer details\n")
        add_file("network", "per_attack_type", "Detection rate per attack type (test period).",
                 "network-attacks")
        add_file("network", "sampling_correction",
                 "Precision corrected for background down-sampling.", "network-sampling", digits=4)

    md.append("## 9. Figures\n")
    for layer in ("transaction", "wallet", "network"):
        src = out_root / layer / "figures"
        if layer in runs and src.exists():
            dst = report_dir / "figures" / layer
            shutil.copytree(src, dst, dirs_exist_ok=True)
            md.append(f"### {LAYER_TITLES[layer]}\n")
            for stem, caption in FIGURES:
                if (dst / f"{stem}.png").exists():
                    md.append(f"![{caption}](figures/{layer}/{stem}.png)\n\n*Figure: {caption} "
                              f"({layer} layer). Vector version: `figures/{layer}/{stem}.pdf`.*\n")

    md.append("## 10. Limitations and threats to validity\n")
    md.append(
        "- **Concept drift.** All layers are trained once and applied to the future. The "
        "transaction layer loses almost all recall after time step 43 (Section 6).\n"
        "- **Label coverage.** Most Elliptic++ transactions and wallets are unlabelled. Metrics "
        "describe the labelled subset; verdicts on unlabelled entities cannot be verified.\n"
        "- **Wallet features look ahead.** Elliptic++ computes each wallet's features over its "
        "whole lifetime, so a wallet's row contains information from after its first appearance.\n"
        "- **Threshold transfer.** Thresholds are fixed on validation steps and can be "
        "mis-calibrated for later periods (see the threshold-sensitivity and calibration figures).\n"
        "- **Layers are not fused.** Elliptic++ and UGR'16 share no entities or timestamps, so the "
        "network layer is an independent detector evaluated on a proxy dataset, not on Bitcoin "
        "network traffic.\n"
        "- **Network sampling.** Background windows are down-sampled; recall and false-positive "
        "rate are unaffected, precision is reported both raw and corrected.\n"
        "- **Single split.** One temporal split per layer; intervals capture test-set sampling "
        "noise and seed variance, not variation across alternative splits.\n")

    (report_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    (report_dir / "tables.tex").write_text(
        "% Requires \\usepackage{booktabs}\n\n" + "\n".join(tex), encoding="utf-8")
    print(f"\nReport written to {report_dir / 'REPORT.md'} ({len(tex)} tables, "
          f"LaTeX in {report_dir / 'tables.tex'})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", default="Dataset_elliptic++")
    p.add_argument("--ugr-input", default="Dataset_UGR",
                   help="UGR'16 flow file or folder for the network layer")
    p.add_argument("--max-flows", type=int, default=None, help="cap on UGR'16 flows read")
    p.add_argument("--background-rate", type=float, default=None,
                   help="fraction of background-only UGR'16 windows kept (script default if unset)")
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--report-dir", default="report")
    p.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), help="run only these")
    p.add_argument("--force", action="store_true", help="re-run finished experiments")
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    if not a.report_only:
        run_experiments(a, ROOT / a.out_root)
    build_report(ROOT / a.out_root, ROOT / a.report_dir)
