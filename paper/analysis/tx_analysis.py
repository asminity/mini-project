"""Supplementary transaction-layer analyses for the manuscript.

A. Static model vs. a model retrained before every test step (prequential evaluation).
B. Random split vs. temporal split.
C. Drift diagnosis with a domain classifier.
D. Isolation Forest by period.
E. Paired bootstrap: ensemble vs. the 50-tree Random Forest of prior work.
F. Label homophily of the transaction graph.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from common import DATA, RESULTS, ROOT, SEED, ap_or_nan, auc_or_nan, core, homophily, lgbm, save, summarise
import txn_anomaly_detection as txn

ROUNDS = 600
core.setup_logging()
ds = txn.load_dataset(DATA)
t, y = ds.time_step, ds.y.astype(int)
lab = y != -1
Xb = core.impute_and_clean(ds.X, t <= 34)
X, names = core.build_graph_features(Xb, ds.edges, ds.feature_names)
n_base = Xb.shape[1]
PERIODS = {"35-42": (35, 42), "43-49": (43, 49), "35-49": (35, 49)}

# ----------------------------------------------------------------- A. prequential
train = lab & (t <= 34)
static = lgbm(SEED, ROUNDS).fit(X[train], y[train])
s_static = static.predict_proba(X)[:, 1]
s_retrain = np.full(len(y), np.nan)
for step in range(35, 50):
    past = lab & (t < step)
    model = lgbm(SEED, ROUNDS).fit(X[past], y[past])
    now = t == step
    s_retrain[now] = model.predict_proba(X[now])[:, 1]
    print(f"[prequential] step {step}: trained on {past.sum()} rows", flush=True)

rows = []
for step in range(35, 50):
    m = lab & (t == step)
    for kind, s in (("static", s_static), ("retrained", s_retrain)):
        r = summarise(y[m], s[m])
        rows.append({"step": step, "model": kind, **r})
save("tx_prequential_per_step", pd.DataFrame(rows))
rows = []
for name, (lo, hi) in PERIODS.items():
    m = lab & (t >= lo) & (t <= hi)
    for kind, s in (("static", s_static), ("retrained", s_retrain)):
        rows.append({"period": name, "model": kind, **summarise(y[m], s[m])})
save("tx_prequential_pooled", pd.DataFrame(rows))

# ------------------------------------------------------------ B. random vs temporal
rows = []
idx = np.flatnonzero(lab)
test_t = lab & (t >= 35)
rf_t = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=SEED).fit(Xb[train], y[train])
s_rf_t = rf_t.predict_proba(Xb)[:, 1]
rows.append({"split": "temporal", "model": "RF-50 (original features)", "seed": SEED,
             **summarise(y[test_t], s_rf_t[test_t])})
rows.append({"split": "temporal", "model": "LightGBM (all features)", "seed": SEED,
             **summarise(y[test_t], s_static[test_t])})
for seed in range(3):
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y[idx])
    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=seed).fit(Xb[tr], y[tr])
    rows.append({"split": "random", "model": "RF-50 (original features)", "seed": seed,
                 **summarise(y[te], rf.predict_proba(Xb[te])[:, 1])})
    gb = lgbm(seed, ROUNDS).fit(X[tr], y[tr])
    rows.append({"split": "random", "model": "LightGBM (all features)", "seed": seed,
                 **summarise(y[te], gb.predict_proba(X[te])[:, 1])})
    print(f"[random split] seed {seed} done", flush=True)
save("tx_random_vs_temporal", pd.DataFrame(rows))

# ------------------------------------------------------------- C. domain classifier
def domain_auc(mask_a, mask_b, cap=4000, seed=SEED):
    rng = np.random.default_rng(seed)
    a, b = np.flatnonzero(mask_a), np.flatnonzero(mask_b)
    a = rng.choice(a, min(cap, len(a)), replace=False)
    b = rng.choice(b, min(cap, len(b)), replace=False)
    Xd = Xb[np.concatenate([a, b])]
    yd = np.r_[np.zeros(len(a)), np.ones(len(b))]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(lgbm(seed, 150), Xd, yd, cv=cv, method="predict_proba")[:, 1]
    return float(auc_or_nan(yd, p)), int(len(a)), int(len(b))

rows = []
for cls, cname in ((1, "illicit"), (0, "licit")):
    c = y == cls
    pairs = {"control: steps 1-17 vs 18-34": (c & (t <= 17), c & (t >= 18) & (t <= 34)),
             "train 1-34 vs test 35-42": (c & (t <= 34), c & (t >= 35) & (t <= 42)),
             "train 1-34 vs test 43-49": (c & (t <= 34), c & (t >= 43)),
             "test 35-42 vs test 43-49": (c & (t >= 35) & (t <= 42), c & (t >= 43))}
    for label, (a, b) in pairs.items():
        auc, na, nb = domain_auc(a, b)
        rows.append({"class": cname, "comparison": label, "n_a": na, "n_b": nb, "domain_auc": auc})
        print(f"[domain] {cname} {label}: AUC {auc:.3f}", flush=True)
# Do post-shutdown illicit transactions look like licit ones? Classifier trained on steps 1-34
# to separate illicit from licit, applied to later illicit transactions.
for label, (lo, hi) in PERIODS.items():
    m_i, m_l = (y == 1) & (t >= lo) & (t <= hi), (y == 0) & (t >= lo) & (t <= hi)
    rows.append({"class": "static-model score", "comparison": f"median score, illicit, steps {label}",
                 "n_a": int(m_i.sum()), "n_b": 0, "domain_auc": float(np.median(s_static[m_i]))})
    rows.append({"class": "static-model score", "comparison": f"median score, licit, steps {label}",
                 "n_a": int(m_l.sum()), "n_b": 0, "domain_auc": float(np.median(s_static[m_l]))})
    rows.append({"class": "static-model score",
                 "comparison": f"share of illicit with score < 0.05, steps {label}",
                 "n_a": int(m_i.sum()), "n_b": 0, "domain_auc": float((s_static[m_i] < 0.05).mean())})
save("tx_drift_diagnosis", pd.DataFrame(rows))

# ---------------------------------------------------------------- D. isolation forest
normal = Xb[train & (y == 0)]
iso = IsolationForest(n_estimators=300, max_samples=min(4096, len(normal)), random_state=SEED,
                      n_jobs=-1).fit(normal)
s_iso = -iso.score_samples(Xb)
rows = []
for label, (lo, hi) in {"1-34 (training period)": (1, 34), **PERIODS}.items():
    m = lab & (t >= lo) & (t <= hi)
    rows.append({"period": label, "n": int(m.sum()), "n_illicit": int(y[m].sum()),
                 "roc_auc": auc_or_nan(y[m], s_iso[m]), "pr_auc": ap_or_nan(y[m], s_iso[m]),
                 "median_score_illicit": float(np.median(s_iso[m & (y == 1)])),
                 "median_score_licit": float(np.median(s_iso[m & (y == 0)]))})
save("tx_isolation_forest", pd.DataFrame(rows))

# ------------------------------------------------------------- E. paired bootstrap
verdicts = pd.read_csv(ROOT / "outputs" / "transaction" / "transaction_verdicts.csv",
                       usecols=["txId", "anomaly_score"]).set_index("txId")
s_ens = verdicts["anomaly_score"].reindex(ds.tx_id).to_numpy()
thr = json.loads((ROOT / "outputs" / "transaction" / "metrics.json").read_text())["decision_threshold"]
yt, a, b = y[test_t] == 1, s_ens[test_t] >= thr, s_rf_t[test_t] >= 0.5


def f1(pos, pred):
    tp = (pos & pred).sum()
    return 2 * tp / max(2 * tp + (~pos & pred).sum() + (pos & ~pred).sum(), 1)


rng = np.random.default_rng(SEED)
n = len(yt)
diffs = []
for _ in range(5000):
    i = rng.integers(0, n, n)
    diffs.append(f1(yt[i], a[i]) - f1(yt[i], b[i]))
diffs = np.array(diffs)
save("tx_paired_bootstrap", {
    "f1_ensemble": float(f1(yt, a)), "f1_rf50": float(f1(yt, b)),
    "difference": float(f1(yt, a) - f1(yt, b)),
    "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
    "share_of_resamples_ensemble_better": float((diffs > 0).mean()), "resamples": 5000,
    "disagreements": int((a != b).sum())})

# ---------------------------------------------------------------------- F. homophily
save("tx_homophily", homophily(ds.edges, ds.y.astype(int)))
print("tx analysis complete", flush=True)
