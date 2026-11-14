"""Supplementary wallet-layer analyses for the manuscript.

A. Feature-group ablation (which graph features carry the gain), 3 seeds.
B. Static model vs. a model retrained before every test step.
C. Random split vs. temporal split, at three row granularities.
D. Performance on wallets with / without look-ahead in their features.
E. Label homophily of the address graph.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from common import DATA, ROOT, SEED, core, homophily, lgbm, save, summarise
import wallet_anomaly_detection as wal

core.setup_logging()
wallets, appearances, edges = wal.load_wallets(DATA)
address = wallets["address"].to_numpy()
first = wallets["first_step"].to_numpy()
y = wallets["label"].to_numpy().astype(int)
base_names = [c for c in wallets.columns if c not in ("address", "first_step", "label")]
n_ts = wallets["num_timesteps_appeared_in"].to_numpy()
Xb = core.impute_and_clean(wallets[base_names].to_numpy(dtype=np.float32), first <= 34)
del wallets
X, names = core.build_graph_features(Xb, edges, base_names)
nb = len(base_names)
lab = y != -1
fit_m, val_m = lab & (first < 30), lab & (first >= 30) & (first <= 34)
train_m, test_m = lab & (first <= 34), lab & (first >= 35)
GROUPS = {
    "original (55)": list(range(nb)),
    "+ structural (9)": list(range(nb + 9)),
    "+ in-neighbour means": list(range(nb)) + list(range(nb + 9, 2 * nb + 9)),
    "+ out-neighbour means": list(range(nb)) + list(range(2 * nb + 9, 3 * nb + 9)),
    "+ both neighbour means": list(range(nb)) + list(range(nb + 9, 3 * nb + 9)),
    "all (174)": list(range(3 * nb + 9)),
    "graph only (119)": list(range(nb, 3 * nb + 9)),
}

# ------------------------------------------------------------------ A. ablation
rows = []
for gname, cols in GROUPS.items():
    Xg = np.ascontiguousarray(X[:, cols])
    for seed in (SEED, SEED + 1, SEED + 2):
        model, n = core.train_lgbm(Xg[fit_m], y[fit_m], Xg[val_m], y[val_m], seed)
        s_val = model.predict_proba(Xg[val_m])[:, 1]
        thr = core.best_f1_threshold(y[val_m], s_val)
        grow = min(train_m.sum() / fit_m.sum(), 1.25)
        final, _ = core.train_lgbm(Xg[train_m], y[train_m], None, None, seed, max(1, round(n * grow)))
        s = final.predict_proba(Xg[test_m])[:, 1]
        rows.append({"features": gname, "n_features": len(cols), "seed": seed, "rounds": n,
                     "threshold": thr, "val_pr_auc": core.evaluate(y[val_m], s_val, thr)["pr_auc"],
                     **summarise(y[test_m], s, thr)})
        print(f"[ablation] {gname} seed {seed}: F1 {rows[-1]['f1']:.3f} AP {rows[-1]['pr_auc']:.3f}",
              flush=True)
    del Xg
save("wallet_feature_ablation", pd.DataFrame(rows))

# ---------------------------------------------------------------- B. prequential
ROUNDS = 80
static = lgbm(SEED, ROUNDS).fit(X[train_m], y[train_m])
s_static = core.proba(static, X)
s_retrain = np.full(len(y), np.nan)
for step in range(35, 50):
    past = lab & (first < step)
    now = first == step
    s_retrain[now] = lgbm(SEED, ROUNDS).fit(X[past], y[past]).predict_proba(X[now])[:, 1]
    print(f"[prequential] step {step}: trained on {past.sum()} wallets", flush=True)
rows, pooled = [], []
for step in range(35, 50):
    m = lab & (first == step)
    for kind, s in (("static", s_static), ("retrained", s_retrain)):
        rows.append({"step": step, "model": kind, **summarise(y[m], s[m])})
for name, (lo, hi) in {"35-42": (35, 42), "43-49": (43, 49), "35-49": (35, 49)}.items():
    m = lab & (first >= lo) & (first <= hi)
    for kind, s in (("static", s_static), ("retrained", s_retrain)):
        pooled.append({"period": name, "model": kind, **summarise(y[m], s[m])})
save("wallet_prequential_per_step", pd.DataFrame(rows))
save("wallet_prequential_pooled", pd.DataFrame(pooled))

# -------------------------------------------------------- D. look-ahead subsets
verdicts = pd.read_csv(ROOT / "outputs" / "wallet" / "wallet_verdicts.csv",
                       usecols=["address", "anomaly_score"]).set_index("address")
s_ens = verdicts["anomaly_score"].reindex(address).to_numpy()
thr = json.loads((ROOT / "outputs" / "wallet" / "metrics.json").read_text())["decision_threshold"]
rows = []
for label, m in (("all test wallets", test_m),
                 ("appear in exactly 1 time step", test_m & (n_ts == 1)),
                 ("appear in 2+ time steps", test_m & (n_ts > 1))):
    rows.append({"subset": label, **summarise(y[m], s_ens[m], thr)})
train_share_single = float((n_ts[train_m] == 1).mean())
save("wallet_lookahead_subsets", pd.DataFrame(rows))
save("wallet_lookahead_meta", {"share_single_step_train": train_share_single,
                              "share_single_step_test": float((n_ts[test_m] == 1).mean())})

# -------------------------------------------------------------------- E. homophily
save("wallet_homophily", homophily(edges, y))

# ------------------------------------------------- C. random vs temporal splits
rows = []
rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=SEED).fit(Xb[train_m], y[train_m])
rows.append({"rows": "one per wallet", "split": "temporal (first-seen)",
             "model": "RF-50 (original features)", "seed": SEED,
             **summarise(y[test_m], rf.predict_proba(Xb[test_m])[:, 1])})
rows.append({"rows": "one per wallet", "split": "temporal (first-seen)",
             "model": "LightGBM (all features)", "seed": SEED,
             **summarise(y[test_m], s_static[test_m])})
idx = np.flatnonzero(lab)
for seed in range(3):
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y[idx])
    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=seed).fit(Xb[tr], y[tr])
    rows.append({"rows": "one per wallet", "split": "random 70/30",
                 "model": "RF-50 (original features)", "seed": seed,
                 **summarise(y[te], rf.predict_proba(Xb[te])[:, 1])})
    gb = lgbm(seed, ROUNDS).fit(X[tr], y[tr])
    rows.append({"rows": "one per wallet", "split": "random 70/30",
                 "model": "LightGBM (all features)", "seed": seed,
                 **summarise(y[te], core.proba(gb, X[te]))})
    print(f"[random split] wallets seed {seed} done", flush=True)
del X

# Rows exactly as distributed: one per (wallet, time step) appearance, including the
# duplicated rows in the file, with the time step as a feature (the published feature set).
raw = pd.read_csv(DATA / "wallets_features_classes_combined.csv")
raw = raw[raw["class"] != 3].reset_index(drop=True)
feat = [c for c in raw.columns if c not in ("address", "class")]
Xr = raw[feat].to_numpy(dtype=np.float32)
yr = (raw["class"] == 1).to_numpy().astype(int)
step = raw["Time step"].to_numpy()
addr = raw["address"].to_numpy()
del raw
tr, te = step <= 34, step > 34
rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=SEED).fit(Xr[tr], yr[tr])
rows.append({"rows": "as distributed (wallet x step, with duplicates)", "split": "temporal (row time step)",
             "model": "RF-50 (original features + time step)", "seed": SEED,
             **summarise(yr[te], rf.predict_proba(Xr[te])[:, 1]),
             "test_rows_with_wallet_in_train": float(np.isin(addr[te], addr[tr]).mean())})
all_idx = np.arange(len(yr))
for seed in range(3):
    a, b = train_test_split(all_idx, test_size=0.3, random_state=seed, stratify=yr)
    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=seed).fit(Xr[a], yr[a])
    rows.append({"rows": "as distributed (wallet x step, with duplicates)", "split": "random 70/30",
                 "model": "RF-50 (original features + time step)", "seed": seed,
                 **summarise(yr[b], rf.predict_proba(Xr[b])[:, 1]),
                 "test_rows_with_wallet_in_train": float(np.isin(addr[b], addr[a]).mean())})
    print(f"[random split] raw rows seed {seed} done", flush=True)
save("wallet_random_vs_temporal", pd.DataFrame(rows))
print("wallet analysis complete", flush=True)
