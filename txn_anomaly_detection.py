"""
Layer 1 - transaction-level anomaly detection on Elliptic++.

Scores every Bitcoin transaction as licit / suspicious / illicit using its
tabular features plus features derived from the transaction graph (degrees,
2-hop reach, component size, mean feature vector of in- and out-neighbours).
The protocol, baselines and reporting are in anomaly_core.py.

Usage
-----
    python txn_anomaly_detection.py --data-dir Dataset_elliptic++ --out-dir outputs/transaction
    python txn_anomaly_detection.py --no-graph-features --out-dir outputs/transaction_no_graph

Needs txs_features.csv, txs_classes.csv and txs_edgelist.csv from
https://github.com/git-disl/EllipticPlusPlus
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import anomaly_core as core
from anomaly_core import ILLICIT, LICIT, UNKNOWN, log


@dataclass
class Dataset:
    tx_id: np.ndarray        # (n,) transaction ids
    time_step: np.ndarray    # (n,) int
    y: np.ndarray            # (n,) 1 illicit, 0 licit, -1 unknown
    X: np.ndarray            # (n, d) float32
    feature_names: list[str]
    edges: np.ndarray        # (m, 2) row indices, src -> dst


def load_dataset(data_dir: Path, use_cache: bool = True) -> Dataset:
    feat_csv = data_dir / "txs_features.csv"
    cache = data_dir / "txs_features.cache.parquet"
    for name in ("txs_features.csv", "txs_classes.csv", "txs_edgelist.csv"):
        if not (data_dir / name).exists():
            raise FileNotFoundError(
                f"{data_dir / name} not found. Download the Elliptic++ transaction files "
                "from https://github.com/git-disl/EllipticPlusPlus into this folder.")

    if use_cache and cache.exists() and cache.stat().st_mtime >= feat_csv.stat().st_mtime:
        feats = pd.read_parquet(cache)
    else:
        log.info("Reading %s (first run; cached to parquet afterwards)", feat_csv.name)
        feats = pd.read_csv(feat_csv)
        if use_cache:
            feats.to_parquet(cache, index=False)

    id_col = core.find_column(feats, {"txid"})
    time_col = core.find_column(feats, {"time step", "timestep"})
    feats = (feats.drop_duplicates(id_col).sort_values([time_col, id_col], kind="stable")
             .reset_index(drop=True))
    feature_names = [c for c in feats.columns if c not in (id_col, time_col)]

    classes = pd.read_csv(data_dir / "txs_classes.csv")
    labels = pd.Series(core.parse_classes(classes[core.find_column(classes, {"class"})]),
                       index=classes[core.find_column(classes, {"txid"})].to_numpy())
    labels = labels[~labels.index.duplicated()]
    y = labels.reindex(feats[id_col].to_numpy()).fillna(UNKNOWN).to_numpy(dtype=np.int8)

    edges_df = pd.read_csv(data_dir / "txs_edgelist.csv")
    index = pd.Index(feats[id_col].to_numpy())
    src = index.get_indexer(edges_df.iloc[:, 0].to_numpy())
    dst = index.get_indexer(edges_df.iloc[:, 1].to_numpy())
    keep = (src >= 0) & (dst >= 0) & (src != dst)
    edges = np.unique(np.column_stack([src[keep], dst[keep]]), axis=0)

    ds = Dataset(tx_id=feats[id_col].to_numpy(),
                 time_step=feats[time_col].to_numpy(dtype=np.int32), y=y,
                 X=feats[feature_names].to_numpy(dtype=np.float32),
                 feature_names=feature_names, edges=edges)
    log.info("Loaded %d transactions, %d features, %d edges | illicit=%d licit=%d unknown=%d",
             len(y), ds.X.shape[1], len(edges),
             (y == ILLICIT).sum(), (y == LICIT).sum(), (y == UNKNOWN).sum())
    return ds


def run(args):
    t0 = time.time()
    ds = load_dataset(Path(args.data_dir), use_cache=not args.no_cache)
    core.check_split(args, ds.time_step)
    X = core.impute_and_clean(ds.X, ds.time_step <= args.train_end)
    names = ds.feature_names
    if not args.no_graph_features:
        X, names = core.build_graph_features(X, ds.edges, names)
    return core.fit_evaluate_report(
        X, names, ds.y, ds.time_step, ds.tx_id, args, id_col="txId", entity="transaction",
        t0=t0, n_base=len(ds.feature_names),
        model_meta={"base_feature_names": ds.feature_names,
                    "graph_features": not args.no_graph_features})


if __name__ == "__main__":
    core.setup_logging()
    parser = core.build_parser(__doc__.split("\n\n")[0])
    parser.set_defaults(out_dir="outputs/transaction")
    run(parser.parse_args())
