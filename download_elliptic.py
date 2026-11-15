"""Download Elliptic++ dataset files directly via GitHub LFS batch API."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

FILES = [
    {
        "name": "txs_classes.csv",
        "oid": "013a11742969071a906878ded0319571df0657f9b7133e5c6cdb36217bf0d240",
        "size": 2361914,
    },
    {
        "name": "txs_edgelist.csv",
        "oid": "a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e",
        "size": 4470584,
    },
    {
        "name": "wallets_classes.csv",
        "oid": "4e5132c99f941666bf1fefd4100a1428d339c9252ec6987909e1adf8eac902f9",
        "size": 30421134,
    },
    {
        "name": "AddrAddr_edgelist.csv",
        "oid": "ffba894458e262a691e5e4d006f5dc1d0e069fabfe828f443fd157bf7f8393f2",
        "size": 200631481,
    },
    {
        "name": "wallets_features.csv",
        "oid": "317daca2810c355ddfdb8c0dab34cf11d1aa90567fe975090c7e5a901386eb77",
        "size": 606463522,
    },
    {
        "name": "txs_features.csv",
        "oid": "2db326ec8ddb68f1d810c1834e1ff62e0a8300378f0984a1e3b2ca82a439821b",
        "size": 694789588,
    },
]

DEST_DIR = Path("Dataset_elliptic++")
BATCH_URL = "https://github.com/git-disl/EllipticPlusPlus.git/info/lfs/objects/batch"


def get_download_url(oid: str, size: int) -> str:
    body = json.dumps({
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": oid, "size": size}],
    }).encode("utf-8")
    req = urllib.request.Request(
        BATCH_URL,
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
            "User-Agent": "git-lfs/3.0.0",
        },
        data=body,
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        obj = data["objects"][0]
        if "error" in obj:
            raise RuntimeError(f"LFS batch error: {obj['error']}")
        return obj["actions"]["download"]["href"]


def download_file(url: str, dest: Path, expected_size: int, expected_sha: str) -> None:
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"[skip] {dest.name} already exists ({expected_size / 1e6:.1f} MB)", flush=True)
        return

    part = dest.with_suffix(dest.suffix + ".part")
    existing_bytes = part.stat().st_size if part.exists() else 0

    headers = {"User-Agent": "git-lfs/3.0.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        print(f"Resuming {dest.name} from {existing_bytes / 1e6:.1f} MB / {expected_size / 1e6:.1f} MB...", flush=True)
    else:
        print(f"Downloading {dest.name} ({expected_size / 1e6:.1f} MB)...", flush=True)

    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    last_log = t0
    sha = hashlib.sha256()

    mode = "ab" if existing_bytes > 0 else "wb"
    with urllib.request.urlopen(req, timeout=60) as resp, open(part, mode) as out:
        downloaded = existing_bytes
        while True:
            chunk = resp.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_log >= 5 or downloaded >= expected_size:
                elapsed = max(now - t0, 0.1)
                speed = (downloaded - existing_bytes) / elapsed / 1e6
                pct = downloaded / expected_size * 100
                print(f"  {dest.name}: {downloaded / 1e6:.1f}/{expected_size / 1e6:.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s", flush=True)
                last_log = now

    part.rename(dest)
    print(f"[done] {dest.name} in {time.time() - t0:.1f}s", flush=True)


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dest = DEST_DIR / f["name"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            print(f"[ok] {f['name']} already verified", flush=True)
            continue
        print(f"Fetching download URL for {f['name']}...", flush=True)
        url = get_download_url(f["oid"], f["size"])
        download_file(url, dest, f["size"], f["oid"])
    print("\nAll Elliptic++ dataset files downloaded successfully!", flush=True)


if __name__ == "__main__":
    main()
