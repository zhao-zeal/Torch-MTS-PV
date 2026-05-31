#!/usr/bin/env python3
"""
Prepare SKIPPD data for reproducing run_solarv4_part1.py inside Torch-MTS-PV.

This script only handles the data layer. It converts the real SKIPPD CSV into
sliding-window arrays with the shape commonly used by Torch-MTS style loaders:

    x: [num_samples, seq_len,  num_nodes, channels]
    y: [num_samples, pred_len, num_nodes, channels]

For strict part1 reproduction, the split is applied BEFORE windowing:
    raw series -> train/val/test by time order -> windows inside each segment

Default target:
    seq_len=672, pred_len=1, train/val/test=60/10/30

Usage:
    python scripts/prepare_part1_skippd_npz.py \
        --csv_file F:/3datas/SKIPPD/skippd.csv \
        --output_npz data/SKIPPD/skippd_part1_sl672_pl1.npz

Or:
    python scripts/prepare_part1_skippd_npz.py \
        --data_dir F:/3datas/SKIPPD \
        --output_npz data/SKIPPD/skippd_part1_sl672_pl1.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def read_csv_with_datetime_index(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if isinstance(df.index, pd.DatetimeIndex):
            return df.sort_index()
    except Exception:
        pass

    df = pd.read_csv(path)
    time_candidates = [
        c for c in df.columns
        if any(k in str(c).lower() for k in ["time", "date", "timestamp"])
    ]
    if not time_candidates:
        raise ValueError(
            f"Cannot find a datetime column in {path}. "
            "Expected the first column, or a column containing time/date/timestamp."
        )
    tcol = time_candidates[0]
    df[tcol] = pd.to_datetime(df[tcol])
    return df.set_index(tcol).sort_index()


def resolve_skippd_csv(data_dir: str | None, csv_file: str | None) -> str:
    if csv_file:
        target = os.path.abspath(csv_file)
        if not os.path.isfile(target):
            raise FileNotFoundError(f"--csv_file does not exist: {target}")
        return target

    if not data_dir:
        raise ValueError("Either --data_dir or --csv_file must be provided.")

    all_csv = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not all_csv:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    exact = [p for p in all_csv if os.path.basename(p).lower() == "skippd.csv"]
    if exact:
        return exact[0]

    named = [p for p in all_csv if "skippd" in os.path.basename(p).lower()]
    if named:
        return named[0]

    preview = "\n".join(f"  - {p}" for p in all_csv[:20])
    raise FileNotFoundError(
        "Cannot find skippd.csv. This script intentionally refuses to use "
        "filename-containing-'15' heuristics because that can select task15.csv.\n"
        "Pass the real file explicitly, for example:\n"
        "  --csv_file F:/3datas/SKIPPD/skippd.csv\n"
        f"Found CSV files include:\n{preview}"
    )


def load_power_series(path: str):
    df = read_csv_with_datetime_index(path)
    df = df[~df.index.duplicated(keep="first")]

    power_col = None
    for col in df.columns:
        cl = str(col).lower()
        if "power" in cl or "pv" in cl:
            power_col = col
            break

    if power_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError(f"No numeric power-like column found in {path}")
        power_col = numeric_cols[0]
        print(f"[WARN] no power/pv column found; using first numeric column: {power_col}")

    power = df[power_col].astype(float).clip(lower=0)
    if len(power) > 150000:
        power = power.resample("15min").mean()
    power = power.ffill().bfill().astype(np.float32)
    return power.values, pd.DatetimeIndex(power.index), str(power_col)


def make_windows(values: np.ndarray, seq_len: int, pred_len: int):
    n = len(values) - seq_len - pred_len + 1
    if n <= 0:
        raise ValueError(
            f"Segment too short for windows: len={len(values)}, "
            f"seq_len={seq_len}, pred_len={pred_len}"
        )

    x = np.empty((n, seq_len, 1, 1), dtype=np.float32)
    y = np.empty((n, pred_len, 1, 1), dtype=np.float32)
    for i in range(n):
        x[i, :, 0, 0] = values[i: i + seq_len]
        y[i, :, 0, 0] = values[i + seq_len: i + seq_len + pred_len]
    return x, y


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SKIPPD part1 NPZ data.")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing skippd.csv.")
    parser.add_argument("--csv_file", type=str, default=None, help="Explicit path to skippd.csv.")
    parser.add_argument("--output_npz", type=str, default="data/SKIPPD/skippd_part1_sl672_pl1.npz")
    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--capacity", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    target = resolve_skippd_csv(args.data_dir, args.csv_file)
    values, timestamps, power_col = load_power_series(target)

    train_idx = int(len(values) * args.train_ratio)
    val_idx = int(len(values) * (args.train_ratio + args.val_ratio))

    train_values = values[:train_idx]
    val_values = values[train_idx:val_idx]
    test_values = values[val_idx:]

    x_train, y_train = make_windows(train_values, args.seq_len, args.pred_len)
    x_val, y_val = make_windows(val_values, args.seq_len, args.pred_len)
    x_test, y_test = make_windows(test_values, args.seq_len, args.pred_len)

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_npz,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )

    meta = {
        "source_file": target,
        "power_col": power_col,
        "output_npz": str(output_npz),
        "capacity": args.capacity,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "raw_length": int(len(values)),
        "time_range": [str(timestamps[0]), str(timestamps[-1])],
        "split_raw_lengths": {
            "train": int(len(train_values)),
            "val": int(len(val_values)),
            "test": int(len(test_values)),
        },
        "window_shapes": {
            "x_train": list(x_train.shape),
            "y_train": list(y_train.shape),
            "x_val": list(x_val.shape),
            "y_val": list(y_val.shape),
            "x_test": list(x_test.shape),
            "y_test": list(y_test.shape),
        },
        "note": "Raw power is stored. Scaling should be done by the LTSF dataloader using train statistics only.",
    }
    meta_path = output_npz.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[Done] SKIPPD part1 data prepared.")
    print(f"  source_file: {target}")
    print(f"  power_col:   {power_col}")
    print(f"  output_npz:  {output_npz}")
    print(f"  meta:        {meta_path}")
    print("  shapes:")
    for k, v in meta["window_shapes"].items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
