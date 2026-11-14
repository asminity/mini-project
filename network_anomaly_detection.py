"""
Layer 3 - network-traffic anomaly detection on UGR'16 NetFlow data.

UGR'16 has one row per network flow. A single flow says little, so flows are
aggregated into behaviour windows: one row per (source IP, time window), with
volume, fan-out, protocol, TCP-flag and port-usage features. Every window ends
with a verdict benign / suspicious / attack. The protocol, baselines
(including an unsupervised Isolation Forest trained on background traffic
only) and reporting are in anomaly_core.py.

The raw files are far too large for memory, so they are streamed in chunks,
straight from the downloaded .tar.gz if you do not want to extract it.
All attack windows are kept; background windows are randomly down-sampled
(--background-rate) and precision is also reported corrected for that.

Usage
-----
    python network_anomaly_detection.py --input Dataset_UGR --out-dir outputs/network
    python network_anomaly_detection.py --input Dataset_UGR/july.week5.csv.tar.gz --max-flows 50000000

--input is a file (.csv, .csv.gz, .tar.gz) or a folder containing such files.
UGR'16 flow format (no header): end time, duration, source IP, destination IP,
source port, destination port, protocol, flags, forwarding status, type of
service, packets, bytes, label.
"""

from __future__ import annotations

import gzip
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import anomaly_core as core
from anomaly_core import ILLICIT, LICIT, UNKNOWN, log

COLUMNS = ["te", "td", "sa", "da", "sp", "dp", "pr", "flg", "fwd", "stos", "pkt", "byt", "label"]
USED = ["te", "td", "sa", "da", "sp", "dp", "pr", "flg", "pkt", "byt", "label"]
SUFFIXES = (".csv", ".csv.gz", ".tar.gz", ".tgz", ".gz")
SUM_COLS = ["n_flows", "pkts", "bytes", "dur_sum", "n_tcp", "n_udp", "n_icmp", "n_syn",
            "n_syn_only", "n_rst", "n_fin", "n_single_pkt", "n_low_port", "n_p22", "n_p23",
            "n_p25", "n_p53", "n_p80", "n_p443", "n_attack", "n_blacklist"]
MAX_COLS = ["dur_max", "n_dst_ips", "n_dst_ports", "n_src_ports"]


# --------------------------------------------------------------------------- #
# Streaming flow reader
# --------------------------------------------------------------------------- #
def resolve_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(f for f in path.iterdir() if f.is_file() and f.name.lower().endswith(SUFFIXES))
        if files:
            return files
    raise FileNotFoundError(
        f"No UGR'16 flow file found at {path}. Put a weekly file (.csv, .csv.gz or .tar.gz) "
        "there; see https://nesg.ugr.es/nesg-ugr16/ for the downloads.")


class _Sequential(io.RawIOBase):
    """Read-only adapter: tar members in streaming mode lack the methods pandas expects."""

    def __init__(self, raw):
        self._raw = raw

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self._raw.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def open_streams(path: Path):
    """Yield binary file objects with CSV content, without extracting archives to disk."""
    name = path.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path, mode="r|*") as tar:
            for member in tar:
                if member.isfile():
                    yield io.BufferedReader(_Sequential(tar.extractfile(member)), 1 << 20)
    elif name.endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            yield fh
    else:
        with open(path, "rb") as fh:
            yield fh


def read_flow_chunks(paths: list[Path], chunk_rows: int, max_flows: int | None):
    read = 0
    for path in paths:
        log.info("Streaming %s", path.name)
        for stream in open_streams(path):
            reader = pd.read_csv(stream, header=None, names=COLUMNS, usecols=USED,
                                 chunksize=chunk_rows, dtype=str, on_bad_lines="skip",
                                 engine="c")
            for chunk in reader:
                if max_flows is not None and read + len(chunk) > max_flows:
                    chunk = chunk.iloc[: max_flows - read]
                read += len(chunk)
                yield chunk
                if max_flows is not None and read >= max_flows:
                    return


# --------------------------------------------------------------------------- #
# Flow -> (source IP, window) aggregation
# --------------------------------------------------------------------------- #
def prepare(chunk: pd.DataFrame, window_s: int) -> pd.DataFrame:
    """Typed per-flow frame with the helper columns the aggregation sums."""
    te = pd.to_datetime(chunk["te"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    ok = te.notna().to_numpy()            # also drops a header row, if the file has one
    chunk, te = chunk[ok], te[ok]
    secs = ((te - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(np.int64)
    num = lambda c: pd.to_numeric(chunk[c], errors="coerce").fillna(0).to_numpy(np.float64)
    dp, pkt = num("dp"), num("pkt")
    proto = chunk["pr"].str.strip().str.upper()
    flags = chunk["flg"].fillna("")
    tcp = (proto == "TCP").to_numpy()
    syn = tcp & flags.str.contains("S", regex=False).to_numpy()
    ack = flags.str.contains("A", regex=False).to_numpy()
    label = chunk["label"].fillna("background").str.strip().str.lower()
    background = (label == "background").to_numpy()
    blacklist = (label == "blacklist").to_numpy()
    return pd.DataFrame({
        "win": secs // window_s * window_s, "sa": chunk["sa"].to_numpy(),
        "da": pd.factorize(chunk["da"])[0], "dp": dp, "sp": num("sp"),
        "pkts": pkt, "bytes": num("byt"), "dur_sum": num("td"), "n_flows": 1,
        "n_tcp": tcp, "n_udp": (proto == "UDP").to_numpy(), "n_icmp": (proto == "ICMP").to_numpy(),
        "n_syn": syn, "n_syn_only": syn & ~ack,
        "n_rst": tcp & flags.str.contains("R", regex=False).to_numpy(),
        "n_fin": tcp & flags.str.contains("F", regex=False).to_numpy(),
        "n_single_pkt": pkt <= 1, "n_low_port": dp < 1024,
        **{f"n_p{p}": dp == p for p in (22, 23, 25, 53, 80, 443)},
        "n_attack": ~background & ~blacklist, "n_blacklist": blacklist,
        "label": label.to_numpy(),
    })


def aggregate(flows: pd.DataFrame) -> pd.DataFrame:
    keys = ["win", "sa"]
    grouped = flows.groupby(keys, sort=False)
    agg = grouped[SUM_COLS].sum()
    agg["dur_max"] = grouped["dur_sum"].max()
    agg["n_dst_ips"] = grouped["da"].nunique()
    agg["n_dst_ports"] = grouped["dp"].nunique()
    agg["n_src_ports"] = grouped["sp"].nunique()
    attacks = flows.loc[flows["n_attack"], keys + ["label"]]
    if len(attacks):
        counts = attacks.groupby(keys + ["label"], sort=False).size().rename("c").reset_index()
        dominant = (counts.sort_values("c").drop_duplicates(keys, keep="last")
                    .set_index(keys)["label"])
        agg["attack_type"] = dominant.reindex(agg.index).fillna("").to_numpy()
    else:
        agg["attack_type"] = ""
    return agg.reset_index()


def build_windows(paths, window_s, background_rate, chunk_rows, max_flows) -> tuple[pd.DataFrame, dict]:
    """Stream the flows and return one row per kept (window, source IP)."""
    t0 = time.time()
    parts, carry = [], None
    stats = {"flows": 0, "windows_total": 0, "labels": {}}

    def keep_sampled(agg: pd.DataFrame) -> pd.DataFrame:
        stats["windows_total"] += len(agg)
        h = pd.util.hash_pandas_object(agg[["win", "sa"]], index=False).to_numpy()
        sampled = (h % np.uint64(1_000_000)) < np.uint64(int(background_rate * 1_000_000))
        return agg[(agg["n_attack"] > 0) | (agg["n_blacklist"] > 0) | sampled]

    for i, chunk in enumerate(read_flow_chunks(paths, chunk_rows, max_flows), start=1):
        flows = prepare(chunk, window_s)
        stats["flows"] += len(flows)
        for k, v in flows["label"].value_counts().items():
            stats["labels"][k] = stats["labels"].get(k, 0) + int(v)
        if carry is not None:
            flows = pd.concat([carry, flows], ignore_index=True)
        # A window can straddle two chunks: hold the newest window back for the next chunk.
        last = flows["win"].max()
        carry = flows[flows["win"] == last]
        done = flows[flows["win"] != last]
        if len(done):
            parts.append(keep_sampled(aggregate(done)))
        if i % 5 == 0:
            log.info("  %.1fM flows read, %.2fM windows kept (%.0fs)", stats["flows"] / 1e6,
                     sum(map(len, parts)) / 1e6, time.time() - t0)
    if carry is not None and len(carry):
        parts.append(keep_sampled(aggregate(carry)))
    if not parts:
        raise ValueError("No flows could be parsed - check that the file is a UGR'16 flow CSV")
    windows = pd.concat(parts, ignore_index=True)

    # Flow files are only roughly time-ordered, so a window can still appear twice. Merge:
    # counts add up; distinct-counts cannot be recovered exactly, so keep the larger one.
    if windows.duplicated(["win", "sa"]).any():
        rules = {**{c: "sum" for c in SUM_COLS}, **{c: "max" for c in MAX_COLS},
                 "attack_type": "max"}
        windows = windows.groupby(["win", "sa"], sort=False).agg(rules).reset_index()
    windows = windows.sort_values(["win", "sa"], kind="stable").reset_index(drop=True)
    log.info("Read %.1fM flows -> %.2fM source-IP windows, kept %d (%.0fs)", stats["flows"] / 1e6,
             stats["windows_total"] / 1e6, len(windows), time.time() - t0)
    return windows, stats


def window_features(w: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    n = w["n_flows"].to_numpy(np.float64)
    f = {
        "n_flows": n, "pkts": w["pkts"], "bytes": w["bytes"],
        "dur_mean": w["dur_sum"] / n, "dur_max": w["dur_max"],
        "n_dst_ips": w["n_dst_ips"], "n_dst_ports": w["n_dst_ports"],
        "n_src_ports": w["n_src_ports"],
        "pkts_per_flow": w["pkts"] / n, "bytes_per_flow": w["bytes"] / n,
        "bytes_per_pkt": w["bytes"] / np.maximum(w["pkts"], 1),
        "dst_ips_per_flow": w["n_dst_ips"] / n, "dst_ports_per_flow": w["n_dst_ports"] / n,
        "src_ports_per_flow": w["n_src_ports"] / n,
        "dst_ports_per_dst_ip": w["n_dst_ports"] / np.maximum(w["n_dst_ips"], 1),
    }
    for c in ("tcp", "udp", "icmp", "syn", "syn_only", "rst", "fin", "single_pkt", "low_port",
              "p22", "p23", "p25", "p53", "p80", "p443"):
        f[f"share_{c}"] = w[f"n_{c}"] / n
    f["syn_only_per_tcp"] = w["n_syn_only"] / np.maximum(w["n_tcp"], 1)
    f["hour_of_day"] = (w["win"] % 86400) // 3600
    return np.column_stack([np.asarray(v, dtype=np.float32) for v in f.values()]), list(f)


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def load_or_build_windows(args, paths) -> tuple[pd.DataFrame, dict]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache, meta_file = out_dir / "windows.cache.parquet", out_dir / "windows.cache.json"
    key = {"inputs": [[p.name, p.stat().st_size] for p in paths], "window": args.window,
           "background_rate": args.background_rate, "max_flows": args.max_flows}
    if not args.no_cache and cache.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        if meta.get("key") == key:
            log.info("Using cached windows from %s", cache)
            return pd.read_parquet(cache), meta["stats"]
    windows, stats = build_windows(paths, args.window, args.background_rate, args.chunk_rows,
                                   args.max_flows)
    if not args.no_cache:
        windows.to_parquet(cache, index=False)
        meta_file.write_text(json.dumps({"key": key, "stats": stats}))
    return windows, stats


def run(args):
    t0 = time.time()
    paths = resolve_inputs(Path(args.input))
    windows, stats = load_or_build_windows(args, paths)

    has_attack = (windows["n_attack"] > 0).to_numpy()
    only_blacklist = ~has_attack & (windows["n_blacklist"] > 0).to_numpy()
    y = np.where(has_attack, ILLICIT, LICIT).astype(np.int8)
    if args.blacklist_as == "exclude":
        y[only_blacklist] = UNKNOWN
    elif args.blacklist_as == "attack":
        y[only_blacklist] = ILLICIT
    log.info("Windows: %d attack, %d background, %d blacklist-only (%s) | flow labels: %s",
             has_attack.sum(), (~has_attack & ~only_blacklist).sum(), only_blacklist.sum(),
             args.blacklist_as, stats["labels"])

    # Time steps for the temporal split: calendar days, or hours when the data is short.
    win = windows["win"].to_numpy()
    unit, step_name = 86400, "day"
    if len(np.unique(win // unit)) < 4:
        unit, step_name = 3600, "hour"
    raw_step = win // unit
    uniq = np.unique(raw_step)
    steps = (np.searchsorted(uniq, raw_step) + 1).astype(np.int32)
    n_steps = len(uniq)
    if n_steps < 3:
        raise ValueError("Need flows spanning at least 3 hours for a temporal train/val/test split")
    if args.train_end is None:
        args.train_end = min(n_steps - 1, max(2, round(0.7 * n_steps)))
    if args.val_start is None:
        args.val_start = max(2, args.train_end - max(1, round(0.15 * n_steps)) + 1)
    log.info("%d %ss of traffic: fit 1-%d, validation %d-%d, test %d-%d", n_steps, step_name,
             args.val_start - 1, args.val_start, args.train_end, args.train_end + 1, n_steps)

    X, names = window_features(windows)
    X = core.impute_and_clean(X, steps <= args.train_end)
    start = pd.to_datetime(win, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    attack_type = windows["attack_type"].to_numpy()
    results, scores = core.fit_evaluate_report(
        X, names, y, steps, windows["sa"].to_numpy(), args, id_col="source_ip", entity="network",
        t0=t0, verdict_names=("benign", "suspicious", "attack"), step_name=step_name,
        extra_cols={"window_start": start, "attack_type": attack_type,
                    "n_flows": windows["n_flows"].to_numpy()},
        model_meta={"window_seconds": args.window, "background_rate": args.background_rate},
        rf_seeds=1)

    # Layer-specific extras: detection rate per attack type, sampling-corrected precision.
    out_dir = Path(args.out_dir)
    test = (scores["split"] == "test") & (scores["true_label"] == "attack")
    by_type = (scores[test].assign(detected=lambda d: d["verdict"] == "attack",
                                   flagged=lambda d: d["verdict"] != "benign")
               .groupby("attack_type")
               .agg(windows=("detected", "size"), detected=("detected", "sum"),
                    detection_rate=("detected", "mean"),
                    detected_or_suspicious_rate=("flagged", "mean"))
               .reset_index().sort_values("windows", ascending=False))
    core.write_table(by_type, out_dir / "tables" / "per_attack_type",
                     "Network layer: detection rate per attack type on the test period.",
                     "tab:network-attack-types")
    m = results["test"]
    fp_full = m["fp"] / args.background_rate
    adj_precision = m["tp"] / max(m["tp"] + fp_full, 1)
    adj = {"background_rate": args.background_rate, "precision_sampled": m["precision"],
           "precision_corrected": adj_precision, "recall": m["recall"],
           "f1_corrected": 2 * adj_precision * m["recall"] / max(adj_precision + m["recall"], 1e-12),
           "fpr": m["fpr"]}
    core.write_table(pd.DataFrame([{"Quantity": k, "Value": v} for k, v in adj.items()]),
                     out_dir / "tables" / "sampling_correction",
                     "Network layer: precision corrected for background down-sampling "
                     "(false positives scaled by 1 / sampling rate).", "tab:network-sampling",
                     digits=4)
    results.update({"sampling_correction": adj, "per_attack_type": by_type.to_dict(orient="records"),
                    "flows_read": stats["flows"], "windows_before_sampling": stats["windows_total"],
                    "flow_labels": stats["labels"], "inputs": [p.name for p in paths]})
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float))

    print("\nDetection rate per attack type (test):")
    print(by_type.to_string(index=False))
    print(f"\nPrecision on sampled background {m['precision']:.3f}; corrected for the "
          f"{args.background_rate:.1%} background sampling {adj_precision:.3f} "
          f"(recall {m['recall']:.3f} and FPR {m['fpr']:.5f} are unaffected)")
    return results, scores


if __name__ == "__main__":
    core.setup_logging()
    parser = core.build_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--input", default="Dataset_UGR",
                        help="UGR'16 flow file (.csv, .csv.gz, .tar.gz) or a folder of them")
    parser.add_argument("--window", type=int, default=60, help="aggregation window in seconds")
    parser.add_argument("--background-rate", type=float, default=0.02,
                        help="fraction of background-only windows kept (attack windows: all)")
    parser.add_argument("--blacklist-as", choices=["exclude", "background", "attack"],
                        default="exclude",
                        help="how to treat windows whose only non-background label is 'blacklist'")
    parser.add_argument("--max-flows", type=int, default=None,
                        help="stop after this many flows (for quick trials)")
    parser.add_argument("--chunk-rows", type=int, default=2_000_000)
    parser.set_defaults(out_dir="outputs/network", n_seeds=2, train_end=None, val_start=None)
    # Many more rows than the Elliptic++ layers: a lighter forest keeps time and RAM in check.
    core.RF_PARAMS.update(n_estimators=300, min_samples_leaf=3)
    run(parser.parse_args())
