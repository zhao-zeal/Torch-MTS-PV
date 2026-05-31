#!/usr/bin/env python3
"""
Reproduce the SKIPPD part of run_solarv4_part1.py.

Important fix:
- Do NOT select files by "15" in filename, because that can incorrectly pick task15.csv.
- Prefer an exact skippd.csv file under --data_dir.
- Use --csv_file when you want to specify the SKIPPD CSV explicitly.

Faithful target:
- single-variable PV power nowcasting
- DLinear
- seq_len=672, pred_len=1
- train/val/test = 60%/10%/30% by time order
- train mean/std normalization
- seed ensemble: 42 and 123
- monthly capacity-normalized MAE/RMSE accuracy

Examples:
    python scripts/reproduce_part1_skippd.py \
        --data_dir /home/zhaopp/workspace/solar-energy/dataset \
        --output_dir outputs/part1_skippd

    python scripts/reproduce_part1_skippd.py \
        --csv_file /home/zhaopp/workspace/solar-energy/dataset/skippd.csv \
        --output_dir outputs/part1_skippd
"""

import argparse
import copy
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_SEEDS = [42, 123]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_csv_with_datetime_index(path: str) -> pd.DataFrame:
    """Read a CSV and make the first column a DatetimeIndex when possible."""
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


def resolve_skippd_csv(data_dir: str | None = None, csv_file: str | None = None) -> str:
    """Resolve the correct SKIPPD CSV file.

    The previous version used a filename-containing-'15' heuristic, which can
    accidentally select task15.csv. For this reproduction we require the real
    SKIPPD series file, preferably named skippd.csv.
    """
    if csv_file:
        target = os.path.abspath(csv_file)
        if not os.path.isfile(target):
            raise FileNotFoundError(f"--csv_file does not exist: {target}")
        return target

    if not data_dir:
        raise ValueError("Either --data_dir or --csv_file must be provided.")

    all_csv = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not all_csv:
        raise FileNotFoundError(f"No CSV files found under SKIPPD directory: {data_dir}")

    # 1) Best case: exact skippd.csv, case-insensitive.
    exact = [f for f in all_csv if os.path.basename(f).lower() == "skippd.csv"]
    if exact:
        return exact[0]

    # 2) Accept close names such as SKIPPD_15min.csv, but still require SKIPPD in filename.
    named = [f for f in all_csv if "skippd" in os.path.basename(f).lower()]
    if named:
        return named[0]

    preview = "\n".join(f"  - {p}" for p in all_csv[:20])
    raise FileNotFoundError(
        "Cannot find skippd.csv under --data_dir. "
        "This script intentionally refuses to fall back to task15.csv.\n"
        "Please pass the real file explicitly, for example:\n"
        "  --csv_file F:/3datas/SKIPPD/skippd.csv\n"
        f"Found CSV files include:\n{preview}"
    )


def load_skippd(data_dir: str | None = None, csv_file: str | None = None):
    """Load SKIPPD power series for the part1 reproduction."""
    target = resolve_skippd_csv(data_dir=data_dir, csv_file=csv_file)
    print(f"[SKIPPD] using file: {target}")

    df = _read_csv_with_datetime_index(target)
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
            raise ValueError(f"No numeric power-like column found in {target}")
        power_col = numeric_cols[0]
        print(f"[WARN] no power/pv column found; use first numeric column: {power_col}")

    power = df[power_col].astype(float).clip(lower=0)

    # Keep the part1 behavior: only resample very long/high-frequency files.
    if len(power) > 150000:
        power = power.resample("15min").mean()

    power = power.ffill().bfill()
    if not isinstance(power.index, pd.DatetimeIndex):
        raise TypeError("Power series index must be DatetimeIndex.")

    return power.values.astype(np.float32), power.index, target, str(power_col)


class PowerDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
        end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
        x_pad = torch.cat([front, x, end], dim=1).unsqueeze(1)
        avg = self.avg(x_pad)
        return avg.squeeze(1)


class DLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, kernel_size: int = 49):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)


def train_one_seed(train_norm, val_norm, seed, args, device):
    set_seed(seed)

    train_dataset = PowerDataset(train_norm, args.seq_len, args.pred_len)
    val_dataset = PowerDataset(val_norm, args.seq_len, args.pred_len)
    if len(train_dataset) <= 0 or len(val_dataset) <= 0:
        raise ValueError(
            "Dataset is too short for the selected seq_len/pred_len. "
            f"train_windows={len(train_dataset)}, val_windows={len(val_dataset)}"
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = DLinear(args.seq_len, args.pred_len, args.kernel_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_model = None
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if epoch == 1 or epoch % args.print_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"    epoch={epoch:03d} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} best={best_loss:.6f} lr={lr:.2e}"
            )

        if patience_count >= args.patience:
            print(f"    early stop at epoch {epoch}; best_val_loss={best_loss:.6f}")
            break

    model.load_state_dict(best_model)
    return model, best_loss


def train_ensemble(train_data, val_data, args, device):
    train_mean = float(train_data.mean())
    train_std = float(train_data.std())
    if train_std < 1e-6:
        train_std = 1.0

    train_norm = (train_data - train_mean) / train_std
    val_norm = (val_data - train_mean) / train_std
    print(f"[Normalize] mean={train_mean:.6f}, std={train_std:.6f}")

    models = []
    seed_losses = []
    for i, seed in enumerate(args.seeds, 1):
        print(f"[Train] seed {seed} ({i}/{len(args.seeds)})")
        model, best_loss = train_one_seed(train_norm, val_norm, seed, args, device)
        models.append(model)
        seed_losses.append({"seed": seed, "best_val_loss": float(best_loss)})
        print(f"[Train] seed {seed} done; best_val_loss={best_loss:.6f}")

    return models, train_mean, train_std, seed_losses


def evaluate_ensemble(models, test_data, test_ts, capacity, train_mean, train_std, args, device):
    test_norm = (test_data - train_mean) / train_std
    test_dataset = PowerDataset(test_norm, args.seq_len, args.pred_len)
    if len(test_dataset) <= 0:
        raise ValueError(
            "Test dataset is too short for the selected seq_len/pred_len. "
            f"test_windows={len(test_dataset)}"
        )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    all_model_preds = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                preds.append(model(x.to(device)).cpu().numpy())
        all_model_preds.append(np.concatenate(preds, axis=0))

    ensemble_pred = np.mean(all_model_preds, axis=0)
    trues = []
    for _, y in test_loader:
        trues.append(y.numpy())
    true_norm = np.concatenate(trues, axis=0)

    pred_power = np.clip(ensemble_pred[:, 0] * train_std + train_mean, 0, None)
    true_power = np.clip(true_norm[:, 0] * train_std + train_mean, 0, None)

    valid_ts = pd.DatetimeIndex(test_ts[args.seq_len: args.seq_len + len(pred_power)])
    if len(valid_ts) != len(pred_power):
        n = min(len(valid_ts), len(pred_power))
        valid_ts = valid_ts[:n]
        pred_power = pred_power[:n]
        true_power = true_power[:n]

    df_pred = pd.DataFrame({
        "timestamp": valid_ts,
        "pred": pred_power,
        "true": true_power,
        "abs_error": np.abs(pred_power - true_power),
        "sq_error": (pred_power - true_power) ** 2,
    })
    df_pred["month"] = pd.to_datetime(df_pred["timestamp"]).dt.to_period("M").astype(str)

    monthly = df_pred.groupby("month").agg(
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
        num_samples=("pred", "size"),
    ).reset_index()
    monthly["mae_acc"] = (1 - monthly["mae"] / capacity) * 100
    monthly["rmse_acc"] = (1 - monthly["rmse"] / capacity) * 100

    metrics = {
        "mae": float(df_pred["abs_error"].mean()),
        "rmse": float(np.sqrt(df_pred["sq_error"].mean())),
        "mae_acc": float(monthly["mae_acc"].mean()),
        "rmse_acc": float(monthly["rmse_acc"].mean()),
        "num_test_windows": int(len(df_pred)),
        "num_months": int(len(monthly)),
    }
    return metrics, monthly, df_pred


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce run_solarv4_part1.py on SKIPPD only.")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to SKIPPD directory. Used to search skippd.csv.")
    parser.add_argument("--csv_file", type=str, default=None, help="Explicit path to skippd.csv. Highest priority.")
    parser.add_argument("--output_dir", type=str, default="outputs/part1_skippd")
    parser.add_argument("--capacity", type=float, default=30.0, help="SKIPPD capacity used for normalized accuracy.")
    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--kernel_size", type=int, default=49)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Config: {json.dumps(vars(args), ensure_ascii=False, indent=2)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    power, timestamps, source_file, power_col = load_skippd(args.data_dir, args.csv_file)
    train_idx = int(len(power) * args.train_ratio)
    val_idx = int(len(power) * (args.train_ratio + args.val_ratio))

    train_data = power[:train_idx]
    val_data = power[train_idx:val_idx]
    test_data = power[val_idx:]
    test_ts = timestamps[val_idx:]

    print(f"[Data] source_file={source_file}")
    print(f"[Data] power_col={power_col}")
    print(f"[Data] length={len(power)} time_range={timestamps[0]} -> {timestamps[-1]}")
    print(f"[Split] train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    models, train_mean, train_std, seed_losses = train_ensemble(train_data, val_data, args, device)
    metrics, monthly, pred_df = evaluate_ensemble(
        models, test_data, test_ts, args.capacity, train_mean, train_std, args, device
    )

    print("\n[Result]")
    print(f"  MAE      = {metrics['mae']:.6f}")
    print(f"  RMSE     = {metrics['rmse']:.6f}")
    print(f"  MAE_ACC  = {metrics['mae_acc']:.2f}%")
    print(f"  RMSE_ACC = {metrics['rmse_acc']:.2f}%")

    monthly_path = out_dir / "part1_skippd_monthly_metrics.csv"
    pred_path = out_dir / "part1_skippd_predictions.csv"
    summary_path = out_dir / "part1_skippd_summary.json"

    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    summary = {
        "script": "scripts/reproduce_part1_skippd.py",
        "source_file": source_file,
        "power_col": power_col,
        "capacity": args.capacity,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "seeds": args.seeds,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "split_lengths": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
        },
        "train_mean": train_mean,
        "train_std": train_std,
        "seed_losses": seed_losses,
        "metrics": metrics,
        "runtime_seconds": time.time() - start_time,
        "outputs": {
            "monthly_metrics": str(monthly_path),
            "predictions": str(pred_path),
            "summary": str(summary_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Saved]")
    print(f"  {monthly_path}")
    print(f"  {pred_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
