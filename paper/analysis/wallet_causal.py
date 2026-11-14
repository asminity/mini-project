"""Does the wallet-level gain from graph features depend on look-ahead through the graph?

The standard features average over ALL neighbours of a wallet, including wallets
that first appear in later time steps (features only, never labels). Here the
aggregation for wallet i is restricted to neighbours j with first_step(j) <=
first_step(i) ("causal" neighbourhood), and the same LightGBM protocol is run.

Also reports simple facts about both graphs used in the manuscript.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import DATA, SEED, core, save, summarise
import txn_anomaly_detection as txn
import wallet_anomaly_detection as wal

core.setup_logging()
facts = {}

# ---- transaction graph: do edges ever cross time steps?
ds = txn.load_dataset(DATA)
ts = ds.time_step
facts["tx_edges"] = int(len(ds.edges))
facts["tx_edges_crossing_time_steps"] = int((ts[ds.edges[:, 0]] != ts[ds.edges[:, 1]]).sum())
deg = np.bincount(ds.edges.ravel(), minlength=len(ts))
facts["tx_isolated_nodes"] = int((deg == 0).sum())
del ds

# ---- wallet graph
wallets, appearances, edges = wal.load_wallets(DATA)
first = wallets["first_step"].to_numpy()
y = wallets["label"].to_numpy().astype(int)
base_names = [c for c in wallets.columns if c not in ("address", "first_step", "label")]
Xb = core.impute_and_clean(wallets[base_names].to_numpy(dtype=np.float32), first <= 34)
del wallets
n, nb = len(y), len(base_names)
src, dst = edges[:, 0], edges[:, 1]
facts["wallet_edges"] = int(len(edges))
facts["wallet_edges_between_different_first_steps"] = int((first[src] != first[dst]).sum())
facts["wallet_edges_train_to_test_period"] = int(((first[src] <= 34) != (first[dst] <= 34)).sum())
train_nodes = first <= 34
has_future_nbr = np.zeros(n, dtype=bool)
has_future_nbr[src[first[dst] > 34]] = True
has_future_nbr[dst[first[src] > 34]] = True
facts["share_train_wallets_with_test_period_neighbour"] = float(has_future_nbr[train_nodes & (y != -1)].mean())
later = np.zeros(n, dtype=bool)
later[src[first[dst] > first[src]]] = True
later[dst[first[src] > first[dst]]] = True
facts["share_labelled_wallets_with_later_neighbour"] = float(later[y != -1].mean())
save("graph_facts", facts)
print(facts, flush=True)


def aggregate(mask_in, mask_out):
    """Neighbour means and degrees using only the edges selected for each direction.

    mask_in  selects edges j->i that may inform i (i = dst);
    mask_out selects edges i->j that may inform i (i = src).
    """
    blocks, names = [Xb], list(base_names)
    for tag, rows, cols in (("in", dst[mask_in], src[mask_in]), ("out", src[mask_out], dst[mask_out])):
        M = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
        d = np.asarray(M.sum(axis=1)).ravel()
        mean = np.asarray(M @ Xb, dtype=np.float32) / np.maximum(d, 1.0)[:, None].astype(np.float32)
        mean[d == 0] = np.nan
        blocks += [d[:, None].astype(np.float32), mean]
        names += [f"{tag}_deg"] + [f"{tag}_nbr_mean__{c}" for c in base_names]
    return np.hstack(blocks), names


everything = np.ones(len(edges), dtype=bool)
variants = {
    "all neighbours (transductive)": (everything, everything),
    "causal neighbours only": (first[src] <= first[dst], first[dst] <= first[src]),
}
lab = y != -1
fit_m, val_m = lab & (first < 30), lab & (first >= 30) & (first <= 34)
train_m, test_m = lab & (first <= 34), lab & (first >= 35)
rows = []
for vname, (m_in, m_out) in variants.items():
    X, _ = aggregate(m_in, m_out)
    kept = (int(m_in.sum()), int(m_out.sum()))
    for seed in (SEED, SEED + 1, SEED + 2):
        model, k = core.train_lgbm(X[fit_m], y[fit_m], X[val_m], y[val_m], seed)
        s_val = model.predict_proba(X[val_m])[:, 1]
        thr = core.best_f1_threshold(y[val_m], s_val)
        grow = min(train_m.sum() / fit_m.sum(), 1.25)
        final, _ = core.train_lgbm(X[train_m], y[train_m], None, None, seed, max(1, round(k * grow)))
        s = final.predict_proba(X[test_m])[:, 1]
        rows.append({"neighbourhood": vname, "edges_in": kept[0], "edges_out": kept[1], "seed": seed,
                     "rounds": k, "threshold": thr, **summarise(y[test_m], s, thr)})
        print(f"[causal] {vname} seed {seed}: F1 {rows[-1]['f1']:.3f} AP {rows[-1]['pr_auc']:.3f}",
              flush=True)
    del X
save("wallet_causal_neighbourhood", pd.DataFrame(rows))
print("causal analysis complete", flush=True)
