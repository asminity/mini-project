"""
Layer 2 - wallet (address / actor) level anomaly detection on Elliptic++.

Scores every wallet as licit / suspicious / illicit using its behavioural
features plus its position in the address-to-address interaction graph. The
protocol, baselines and reporting are in anomaly_core.py.

Two evaluation protocols
------------------------
first-seen (default, leak-free)
    One row per wallet, placed at the time step where the wallet first appears.
    Train on wallets first seen in steps 1-34, test on wallets first seen in
    35-49, so no test wallet was ever seen in training.
paper
    One row per (wallet, time step) appearance, split on that time step. This
    mirrors the row layout of wallets_features.csv. A wallet's features are
    identical on every row, so wallets active on both sides of the split are
    in train and test at once; use it only to compare with published numbers.

Usage
-----
    python wallet_anomaly_detection.py --data-dir Dataset_elliptic++ --out-dir outputs/wallet
    python wallet_anomaly_detection.py --protocol paper --out-dir outputs/wallet_paper_protocol

Needs wallets_features.csv, wallets_classes.csv and AddrAddr_edgelist.csv.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import anomaly_core as core
from anomaly_core import UNKNOWN, log

FILES = ("wallets_features.csv", "wallets_classes.csv", "AddrAddr_edgelist.csv")


def load_wallets(data_dir: Path, use_cache: bool = True):
    """Returns (wallet table, appearances, edges).

    wallet table : one row per address, sorted by address, with first_step,
                   label and the feature columns.
    appearances  : (k, 2) unique [wallet row index, time step] pairs.
    edges        : (m, 2) unique [sender index, receiver index] pairs.
    """
    for name in FILES:
        if not (data_dir / name).exists():
            raise FileNotFoundError(
                f"{data_dir / name} not found. Download the Elliptic++ actor files "
                "from https://github.com/git-disl/EllipticPlusPlus into this folder."
            )
    table_cache = data_dir / "wallets.cache.parquet"
    graph_cache = data_dir / "wallets_graph.cache.npz"
    newest_src = max((data_dir / f).stat().st_mtime for f in FILES)
    if (use_cache and table_cache.exists() and graph_cache.exists()
            and min(table_cache.stat().st_mtime, graph_cache.stat().st_mtime) >= newest_src):
        graph = np.load(graph_cache)
        return pd.read_parquet(table_cache), graph["appearances"], graph["edges"]

    log.info("Reading wallet csv files (first run; cached afterwards)")
    raw = pd.read_csv(data_dir / "wallets_features.csv")
    addr_col = core.find_column(raw, {"address"})
    time_col = core.find_column(raw, {"time step", "timestep"})
    feature_names = [c for c in raw.columns if c not in (addr_col, time_col)]

    # Features are per wallet (identical on each of its rows): keep one row each.
    wallets = (raw.drop_duplicates(addr_col).drop(columns=time_col)
               .sort_values(addr_col, kind="stable").reset_index(drop=True))
    wallets = wallets.rename(columns={addr_col: "address"})
    wallets[feature_names] = wallets[feature_names].astype(np.float32)
    index = pd.Index(wallets["address"])

    pairs = raw[[addr_col, time_col]].drop_duplicates()
    appearances = np.column_stack([index.get_indexer(pairs[addr_col]),
                                   pairs[time_col].to_numpy()]).astype(np.int32)
    del raw
    first = pd.Series(appearances[:, 1]).groupby(appearances[:, 0]).min()
    wallets.insert(1, "first_step", first.reindex(range(len(wallets))).to_numpy(dtype=np.int32))

    classes = pd.read_csv(data_dir / "wallets_classes.csv")
    labels = pd.Series(core.parse_classes(classes[core.find_column(classes, {"class"})]),
                       index=classes[core.find_column(classes, {"address"})].to_numpy())
    labels = labels[~labels.index.duplicated()]
    wallets.insert(2, "label", labels.reindex(wallets["address"]).fillna(UNKNOWN)
                   .to_numpy(dtype=np.int8))

    edge_df = pd.read_csv(data_dir / "AddrAddr_edgelist.csv")
    src = index.get_indexer(edge_df.iloc[:, 0])
    dst = index.get_indexer(edge_df.iloc[:, 1])
    keep = (src >= 0) & (dst >= 0) & (src != dst)
    edges = np.unique(np.column_stack([src[keep], dst[keep]]), axis=0).astype(np.int32)

    if use_cache:
        wallets.to_parquet(table_cache, index=False)
        np.savez(graph_cache, appearances=appearances, edges=edges)
    return wallets, appearances, edges


def run(args) -> dict:
    t0 = time.time()
    wallets, appearances, edges = load_wallets(Path(args.data_dir), use_cache=not args.no_cache)
    address = wallets["address"].to_numpy()
    first_step = wallets["first_step"].to_numpy()
    y = wallets["label"].to_numpy()
    base_names = [c for c in wallets.columns if c not in ("address", "first_step", "label")]
    log.info("Loaded %d wallets, %d features, %d edges | illicit=%d licit=%d unknown=%d",
             len(wallets), len(base_names), len(edges),
             (y == core.ILLICIT).sum(), (y == core.LICIT).sum(), (y == UNKNOWN).sum())

    core.check_split(args, first_step)
    X = core.impute_and_clean(wallets[base_names].to_numpy(dtype=np.float32),
                              first_step <= args.train_end)
    del wallets
    names = base_names
    if not args.no_graph_features:
        X, names = core.build_graph_features(X, edges, base_names)

    if args.protocol == "paper":
        rows, steps = appearances[:, 0], appearances[:, 1]
        seen_in_train = np.isin(rows, rows[steps <= args.train_end])
        test_lab = (steps > args.train_end) & (y[rows] != UNKNOWN)
        log.info("paper protocol: %.1f%% of labelled test rows are wallets also present in training",
                 100 * seen_in_train[test_lab].mean())
        X, y, address = X[rows], y[rows], address[rows]
    else:
        steps = first_step

    return core.fit_evaluate_report(
        X, names, y, steps, address, args, id_col="address", entity="wallet", t0=t0,
        n_base=len(base_names), model_meta={"base_feature_names": base_names, "protocol": args.protocol,
                    "graph_features": not args.no_graph_features},
        rf_seeds=1,
    )


if __name__ == "__main__":
    core.setup_logging()
    # ~6x more labelled rows than the transaction task: lighter forest keeps RAM in check.
    core.RF_PARAMS.update(n_estimators=300, min_samples_leaf=2)
    parser = core.build_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--protocol", choices=["first-seen", "paper"], default="first-seen",
                        help="evaluation protocol (see module docstring)")
    parser.set_defaults(out_dir="outputs/wallet", n_seeds=2)
    run(parser.parse_args())
