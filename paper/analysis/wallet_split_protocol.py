"""Wallet layer: random vs. temporal split at two row granularities.

(Split out of wallet_analysis.py, whose last stage was interrupted; same logic,
loaded in two stages to keep peak memory low.)
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from common import DATA, SEED, core, lgbm, save, summarise
import wallet_anomaly_detection as wal

core.setup_logging()
ROUNDS = 80
rows = []

# ---- one row per wallet
wallets, appearances, edges = wal.load_wallets(DATA)
first = wallets["first_step"].to_numpy()
y = wallets["label"].to_numpy().astype(int)
base_names = [c for c in wallets.columns if c not in ("address", "first_step", "label")]
Xb = core.impute_and_clean(wallets[base_names].to_numpy(dtype=np.float32), first <= 34)
del wallets, appearances
X, _ = core.build_graph_features(Xb, edges, base_names)
lab = y != -1
train_m, test_m = lab & (first <= 34), lab & (first >= 35)

rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=SEED).fit(Xb[train_m], y[train_m])
rows.append({"rows": "one per wallet", "split": "temporal (first-seen)",
             "model": "RF-50 (original features)", "seed": SEED,
             **summarise(y[test_m], rf.predict_proba(Xb[test_m])[:, 1])})
gb = lgbm(SEED, ROUNDS).fit(X[train_m], y[train_m])
rows.append({"rows": "one per wallet", "split": "temporal (first-seen)",
             "model": "LightGBM (all features)", "seed": SEED,
             **summarise(y[test_m], core.proba(gb, X[test_m]))})
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
del X, Xb, rf, gb
gc.collect()

# ---- rows exactly as distributed (wallet x time step, duplicates included), time step as a feature
raw = pd.read_csv(DATA / "wallets_features_classes_combined.csv")
raw = raw[raw["class"] != 3].reset_index(drop=True)
feat = [c for c in raw.columns if c not in ("address", "class")]
Xr = raw[feat].to_numpy(dtype=np.float32)
yr = (raw["class"] == 1).to_numpy().astype(int)
step = raw["Time step"].to_numpy()
addr = pd.factorize(raw["address"])[0]
del raw
gc.collect()
tr, te = step <= 34, step > 34
rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=SEED).fit(Xr[tr], yr[tr])
rows.append({"rows": "as distributed", "split": "temporal (row time step)",
             "model": "RF-50 (original features + time step)", "seed": SEED,
             **summarise(yr[te], rf.predict_proba(Xr[te])[:, 1]),
             "test_rows_with_wallet_in_train": float(np.isin(addr[te], addr[tr]).mean())})
all_idx = np.arange(len(yr))
for seed in range(3):
    a, b = train_test_split(all_idx, test_size=0.3, random_state=seed, stratify=yr)
    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=seed).fit(Xr[a], yr[a])
    rows.append({"rows": "as distributed", "split": "random 70/30",
                 "model": "RF-50 (original features + time step)", "seed": seed,
                 **summarise(yr[b], rf.predict_proba(Xr[b])[:, 1]),
                 "test_rows_with_wallet_in_train": float(np.isin(addr[b], addr[a]).mean())})
    print(f"[random split] raw rows seed {seed} done", flush=True)
save("wallet_random_vs_temporal", pd.DataFrame(rows))
print("wallet split analysis complete", flush=True)
