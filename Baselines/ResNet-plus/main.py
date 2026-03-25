from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def rmse_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape_np(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def mape_loss(y_pred, y_true, eps=1e-8):
    denom = torch.clamp(torch.abs(y_true), min=eps)
    return torch.mean(torch.abs((y_true - y_pred) / denom)) * 100.0


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_tabular(file_path: str) -> pd.DataFrame:
    fp = str(file_path)
    fp_l = fp.lower()

    if fp_l.endswith(".csv"):
        return pd.read_csv(fp)

    if fp_l.endswith(".npz"):
        z = np.load(fp, allow_pickle=True)
        if "records" in z:
            return pd.DataFrame.from_records(z["records"])
        if "data" in z and "columns" in z:
            data = z["data"]
            cols = [str(c) for c in z["columns"].tolist()]
            return pd.DataFrame(data, columns=cols)
        raise ValueError(f"Invalid npz format: {fp}. Expected 'records' or ('data' and 'columns').")

    raise ValueError(f"Unsupported file type: {fp}. Use .csv or .npz")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_file: str


DATASET_TABLE: Dict[str, DatasetConfig] = {
    "Gympie": DatasetConfig(name="Gympie", data_file="Gym.npz"),
    "Coolum": DatasetConfig(name="Coolum", data_file="Coo.npz"),
    "Noosaville": DatasetConfig(name="Noosaville", data_file="Noo.npz"),
    "Tewantin": DatasetConfig(name="Tewantin", data_file="Tew.npz"),
}


@dataclass
class TrainConfig:
    datasets: List[str]
    data_root: str
    save_root: str

    his_len: int = 6
    pre_len: int = 3

    add_weather_noise: bool = True
    noise_seed: int = 42

    train_end_date: str = "2021-11-30"
    val_start_date: str = "2021-12-01"
    val_end_date: str = "2022-02-10"
    test_start_date: str = "2022-02-11"
    test_end_date: str = "2022-03-04"

    hidden_dim: int = 20
    num_res_blocks: int = 7

    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 300
    early_stopping_patience: int = 30

    random_state: int = 42
    device: str = "cuda"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--datasets",
        nargs="+",
        default=["Gympie"],
        choices=list(DATASET_TABLE.keys()),
    )
    p.add_argument("--data_root", type=str, default=r"D:\python project\image extreme weather\data")
    p.add_argument("--save_root", type=str, default=r"D:\python project\image extreme weather\resnetplus_torch_v2_results")

    p.add_argument("--his_len", type=int, default=6)
    p.add_argument("--pre_len", type=int, default=3)

    p.add_argument("--add_weather_noise", action="store_true", default=True)
    p.add_argument("--no_weather_noise", dest="add_weather_noise", action="store_false")
    p.add_argument("--noise_seed", type=int, default=42)

    p.add_argument("--train_end_date", type=str, default="2021-11-30")
    p.add_argument("--val_start_date", type=str, default="2021-12-01")
    p.add_argument("--val_end_date", type=str, default="2022-02-10")
    p.add_argument("--test_start_date", type=str, default="2022-02-11")
    p.add_argument("--test_end_date", type=str, default="2022-03-04")

    p.add_argument("--hidden_dim", type=int, default=20)
    p.add_argument("--num_res_blocks", type=int, default=7)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--early_stopping_patience", type=int, default=30)

    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    return p


def make_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        datasets=args.datasets,
        data_root=args.data_root,
        save_root=args.save_root,
        his_len=args.his_len,
        pre_len=args.pre_len,
        add_weather_noise=args.add_weather_noise,
        noise_seed=args.noise_seed,
        train_end_date=args.train_end_date,
        val_start_date=args.val_start_date,
        val_end_date=args.val_end_date,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        hidden_dim=args.hidden_dim,
        num_res_blocks=args.num_res_blocks,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        random_state=args.random_state,
        device=args.device,
    )


def build_samples_from_file(
    file_path: str,
    his_length: int,
    pre_length: int,
    add_weather_noise: bool,
    noise_seed: int,
    train_end_date: str,
    val_start_date: str,
    val_end_date: str,
    test_start_date: str,
    test_end_date: str | None,
):
    data = _read_tabular(file_path)

    row_ts = pd.to_datetime(data[["year", "month", "day", "hour"]])

    train_end_ts = pd.Timestamp(train_end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    val_start_ts = pd.Timestamp(val_start_date)
    val_end_ts = pd.Timestamp(val_end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    test_start_ts = pd.Timestamp(test_start_date)
    test_end_ts = None if test_end_date is None else (
        pd.Timestamp(test_end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    )

    train_row_mask = row_ts <= train_end_ts

    load = data.iloc[:, 0].values.astype(np.float32)
    scaler_y = float(np.max(load[train_row_mask]))
    if scaler_y <= 0:
        raise ValueError("Training load max <= 0, normalization invalid.")
    load_norm = load / scaler_y

    weather_cols = ["tempC", "windspeedKmph", "precipMM", "humidity"]
    for c in weather_cols:
        if c not in data.columns:
            raise ValueError(f"Missing weather column: {c}")

    time_cols = [c for c in data.columns if c.startswith("hour_") or c.startswith("weekday_")]
    if not time_cols:
        raise ValueError("No time one-hot columns found.")

    daily_cols = [f"load_{k}_days_ago" for k in range(1, 7)]
    weekly_cols = [f"load_{k}_weeks_ago" for k in range(1, 5)]

    for c in daily_cols + weekly_cols:
        if c not in data.columns:
            raise ValueError(f"Missing lag column: {c}")

    weather_raw = data[weather_cols].values.astype(np.float32)
    time_raw = data[time_cols].values.astype(np.float32)
    daily_raw = data[daily_cols].values.astype(np.float32)
    weekly_raw = data[weekly_cols].values.astype(np.float32)

    rng = np.random.default_rng(noise_seed)
    precip_cv_1to9 = np.array([0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.25, 1.30], dtype=np.float32)

    def _interp_sigma(lead, s1, s9):
        lead = int(np.clip(lead, 1, 9))
        return s1 + (lead - 1) * (s9 - s1) / 8.0

    weather_max = np.maximum(weather_raw[train_row_mask].max(axis=0), 1e-6)
    weather = weather_raw / weather_max

    time_max = np.maximum(time_raw[train_row_mask].max(axis=0), 1e-6)
    time = time_raw / time_max

    daily = daily_raw / scaler_y
    weekly = weekly_raw / scaler_y

    n = len(load_norm)
    if n < his_length + pre_length:
        raise ValueError("Insufficient data length.")

    X_past = []
    X_daily = []
    X_weekly = []
    X_weather = []
    X_time = []
    Y = []
    ts_start = []

    for i in range(his_length, n - pre_length + 1):
        x_past = load_norm[i - his_length:i]
        x_daily = daily[i]
        x_weekly = weekly[i]

        if add_weather_noise:
            w = weather_raw[i:i + pre_length].copy()
            for k in range(pre_length):
                lead = k + 1
                w[k, 0] += rng.normal(0, _interp_sigma(lead, 0.5, 2.0))
                w[k, 1] += rng.normal(0, _interp_sigma(lead, 1.0, 2.0) * 3.6)
                w[k, 3] += rng.normal(0, _interp_sigma(lead, 5.0, 12.0))
                w[k, 2] *= 1.0 + rng.normal(0, precip_cv_1to9[min(lead, 9) - 1])

            w[:, 1] = np.clip(w[:, 1], 0, None)
            w[:, 2] = np.clip(w[:, 2], 0, None)
            w[:, 3] = np.clip(w[:, 3], 0, 100)
            x_weather = w / weather_max
        else:
            x_weather = weather[i:i + pre_length]

        x_time = time[i:i + pre_length]
        y = load_norm[i:i + pre_length]

        X_past.append(x_past)
        X_daily.append(x_daily)
        X_weekly.append(x_weekly)
        X_weather.append(x_weather)
        X_time.append(x_time)
        Y.append(y)
        ts_start.append(row_ts.iloc[i])

    X_past = np.asarray(X_past, dtype=np.float32)
    X_daily = np.asarray(X_daily, dtype=np.float32)
    X_weekly = np.asarray(X_weekly, dtype=np.float32)
    X_weather = np.asarray(X_weather, dtype=np.float32)
    X_time = np.asarray(X_time, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    ts_start = pd.to_datetime(ts_start)
    ts_end = ts_start + pd.to_timedelta(pre_length - 1, unit="h")

    train_mask = ts_end <= train_end_ts
    val_mask = (ts_start >= val_start_ts) & (ts_end <= val_end_ts)
    test_mask = ts_start >= test_start_ts if test_end_ts is None else (
        (ts_start >= test_start_ts) & (ts_end <= test_end_ts)
    )

    split = {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "ts_start": ts_start,
        "ts_end": ts_end,
    }

    meta = {
        "scaler_y": scaler_y,
        "weather_cols": weather_cols,
        "time_cols": time_cols,
        "daily_cols": daily_cols,
        "weekly_cols": weekly_cols,
        "past_dim": int(X_past.shape[1]),
        "daily_dim": int(X_daily.shape[1]),
        "weekly_dim": int(X_weekly.shape[1]),
        "weather_dim": int(X_weather.shape[2]),
        "time_dim": int(X_time.shape[2]),
        "pre_len": int(pre_length),
    }

    features = {
        "past": X_past,
        "daily": X_daily,
        "weekly": X_weekly,
        "weather": X_weather,
        "time": X_time,
    }

    return features, Y, split, meta


class TwoLayerBranch(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        return x


class RecentBranch(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = None
        self.fc2 = None

    def _build(self, in_dim, device):
        self.fc1 = nn.Linear(in_dim, self.hidden_dim).to(device)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim).to(device)

    def forward(self, x):
        if self.fc1 is None or self.fc1.in_features != x.shape[1]:
            self._build(x.shape[1], x.device)
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        return x


class HorizonBasicStructureV2(nn.Module):
    def __init__(self, past_dim, daily_dim, weekly_dim, weather_dim, time_dim, hidden_dim):
        super().__init__()
        self.past_dim = past_dim
        self.hidden_dim = hidden_dim

        self.past_branch = RecentBranch(hidden_dim)
        self.daily_branch = TwoLayerBranch(daily_dim, hidden_dim)
        self.weekly_branch = TwoLayerBranch(weekly_dim, hidden_dim)
        self.weather_branch = TwoLayerBranch(weather_dim, hidden_dim)
        self.time_branch = TwoLayerBranch(time_dim, hidden_dim)

        self.fc_dw1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_dw2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_wt1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_wt2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_global1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc_global2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out1 = nn.Linear(hidden_dim * 2 + 1, hidden_dim)
        self.fc_out2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_pred = nn.Linear(hidden_dim, 1)

    def forward(self, past, daily, weekly, weather_h, time_h, prev_outputs):
        if prev_outputs is None or prev_outputs.shape[1] == 0:
            recent = past
        else:
            recent = torch.cat([past, prev_outputs], dim=1)

        h_past = self.past_branch(recent)
        h_daily = self.daily_branch(daily)
        h_weekly = self.weekly_branch(weekly)
        h_weather = self.weather_branch(weather_h)
        h_time = self.time_branch(time_h)

        h_dw = F.selu(self.fc_dw1(torch.cat([h_daily, h_weather], dim=1)))
        h_dw = F.selu(self.fc_dw2(h_dw))

        h_wt = F.selu(self.fc_wt1(torch.cat([h_weekly, h_time], dim=1)))
        h_wt = F.selu(self.fc_wt2(h_wt))

        h_global = F.selu(self.fc_global1(torch.cat([h_dw, h_wt, h_past], dim=1)))
        h_global = F.selu(self.fc_global2(h_global))

        aux_scalar = weather_h[:, :1]
        h = F.selu(self.fc_out1(torch.cat([h_global, h_past, aux_scalar], dim=1)))
        h = F.selu(self.fc_out2(h))
        out = self.fc_pred(h)
        return out


class ResidualVectorBlock(nn.Module):
    def __init__(self, vec_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(vec_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vec_dim)

    def forward(self, x):
        return x + self.fc2(F.selu(self.fc1(x)))


class ResNetPlusTorchV2(nn.Module):
    def __init__(
        self,
        pre_len,
        past_dim,
        daily_dim,
        weekly_dim,
        weather_dim,
        time_dim,
        hidden_dim=20,
        num_res_blocks=7,
    ):
        super().__init__()
        self.pre_len = pre_len
        self.horizon_blocks = nn.ModuleList([
            HorizonBasicStructureV2(
                past_dim=past_dim,
                daily_dim=daily_dim,
                weekly_dim=weekly_dim,
                weather_dim=weather_dim,
                time_dim=time_dim,
                hidden_dim=hidden_dim,
            )
            for _ in range(pre_len)
        ])
        self.res_blocks = nn.ModuleList([
            ResidualVectorBlock(pre_len, hidden_dim)
            for _ in range(num_res_blocks)
        ])

    def forward(self, past, daily, weekly, weather, time_feat):
        outputs = []
        prev = None
        for h in range(self.pre_len):
            out_h = self.horizon_blocks[h](
                past=past,
                daily=daily,
                weekly=weekly,
                weather_h=weather[:, h, :],
                time_h=time_feat[:, h, :],
                prev_outputs=prev,
            )
            outputs.append(out_h)
            prev = torch.cat(outputs, dim=1)

        coarse = torch.cat(outputs, dim=1)
        refined = coarse
        for blk in self.res_blocks:
            refined = blk(refined)
        return refined


def make_loader(features, targets, mask, batch_size, shuffle):
    past = torch.from_numpy(features["past"][mask]).float()
    daily = torch.from_numpy(features["daily"][mask]).float()
    weekly = torch.from_numpy(features["weekly"][mask]).float()
    weather = torch.from_numpy(features["weather"][mask]).float()
    time_feat = torch.from_numpy(features["time"][mask]).float()
    y = torch.from_numpy(targets[mask]).float()

    ds = TensorDataset(past, daily, weekly, weather, time_feat, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate_model(model, loader, device):
    model.eval()
    preds = []
    trues = []
    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for past, daily, weekly, weather, time_feat, y in loader:
            past = past.to(device)
            daily = daily.to(device)
            weekly = weekly.to(device)
            weather = weather.to(device)
            time_feat = time_feat.to(device)
            y = y.to(device)

            pred = model(past, daily, weekly, weather, time_feat)
            loss = mape_loss(pred, y)

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_n += bs

            preds.append(pred.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return total_loss / max(total_n, 1), preds, trues


def train_model(model, train_loader, val_loader, cfg: TrainConfig, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for past, daily, weekly, weather, time_feat, y in train_loader:
            past = past.to(device)
            daily = daily.to(device)
            weekly = weekly.to(device)
            weather = weather.to(device)
            time_feat = time_feat.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(past, daily, weekly, weather, time_feat)
            loss = mape_loss(pred, y)
            loss.backward()
            optimizer.step()

        val_loss, _, _ = evaluate_model(model, val_loader, device)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    return model, best_val, best_epoch


def evaluate_multistep(y_true_norm: np.ndarray, y_pred_norm: np.ndarray, scaler_y: float):
    y_true = y_true_norm * scaler_y
    y_pred = y_pred_norm * scaler_y

    overall = {
        "MAPE": mape_np(y_true.reshape(-1), y_pred.reshape(-1)),
        "MAE": mae_np(y_true.reshape(-1), y_pred.reshape(-1)),
        "RMSE": rmse_np(y_true.reshape(-1), y_pred.reshape(-1)),
    }

    per_horizon = []
    for h in range(y_true.shape[1]):
        yh_true = y_true[:, h]
        yh_pred = y_pred[:, h]
        per_horizon.append({
            "horizon": h + 1,
            "MAPE": mape_np(yh_true, yh_pred),
            "MAE": mae_np(yh_true, yh_pred),
            "RMSE": rmse_np(yh_true, yh_pred),
        })

    return overall, per_horizon


def run_one_dataset(ds_cfg: DatasetConfig, cfg: TrainConfig):
    data_root = Path(cfg.data_root)
    save_root = Path(cfg.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    data_path = data_root / ds_cfg.data_file
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    print(f"\n================ Dataset: {ds_cfg.name} ================")
    print(f"data_path : {data_path}")
    print(f"device    : {device}")
    print("========================================================")

    t0 = time.time()

    features, targets, split, meta = build_samples_from_file(
        file_path=str(data_path),
        his_length=cfg.his_len,
        pre_length=cfg.pre_len,
        add_weather_noise=cfg.add_weather_noise,
        noise_seed=cfg.noise_seed,
        train_end_date=cfg.train_end_date,
        val_start_date=cfg.val_start_date,
        val_end_date=cfg.val_end_date,
        test_start_date=cfg.test_start_date,
        test_end_date=cfg.test_end_date,
    )

    train_mask = split["train_mask"]
    val_mask = split["val_mask"]
    test_mask = split["test_mask"]

    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            f"Empty split detected: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}"
        )

    train_loader = make_loader(features, targets, train_mask, cfg.batch_size, True)
    val_loader = make_loader(features, targets, val_mask, cfg.batch_size, False)
    test_loader = make_loader(features, targets, test_mask, cfg.batch_size, False)

    model = ResNetPlusTorchV2(
        pre_len=cfg.pre_len,
        past_dim=meta["past_dim"],
        daily_dim=meta["daily_dim"],
        weekly_dim=meta["weekly_dim"],
        weather_dim=meta["weather_dim"],
        time_dim=meta["time_dim"],
        hidden_dim=cfg.hidden_dim,
        num_res_blocks=cfg.num_res_blocks,
    ).to(device)

    print(f"Train: {int(train_mask.sum())} | Val: {int(val_mask.sum())} | Test: {int(test_mask.sum())}")
    print(f"Train range: <= {cfg.train_end_date}")
    print(f"Val range  : {cfg.val_start_date} to {cfg.val_end_date}")
    print(f"Test range : {cfg.test_start_date} to {cfg.test_end_date}")

    model, best_val, best_epoch = train_model(model, train_loader, val_loader, cfg, device)

    _, y_pred_test, y_true_test = evaluate_model(model, test_loader, device)

    overall_metrics, per_horizon_metrics = evaluate_multistep(
        y_true_norm=y_true_test,
        y_pred_norm=y_pred_test,
        scaler_y=meta["scaler_y"],
    )

    print(f"\nBest val MAPE: {best_val:.6f} @ epoch {best_epoch}")
    print("\n[Test Overall]")
    print(f"MAPE: {overall_metrics['MAPE']:.6f}")
    print(f"MAE : {overall_metrics['MAE']:.6f}")
    print(f"RMSE: {overall_metrics['RMSE']:.6f}")

    print("\n[Test Per Horizon]")
    for item in per_horizon_metrics:
        print(
            f"h={item['horizon']:02d} | "
            f"MAPE={item['MAPE']:.6f} | "
            f"MAE={item['MAE']:.6f} | "
            f"RMSE={item['RMSE']:.6f}"
        )

    out_dir = save_root / ds_cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "dataset": ds_cfg.name,
        "config": asdict(cfg),
        "meta": meta,
        "num_train": int(train_mask.sum()),
        "num_val": int(val_mask.sum()),
        "num_test": int(test_mask.sum()),
        "best_val_mape": float(best_val),
        "best_epoch": int(best_epoch),
        "overall_metrics": overall_metrics,
        "per_horizon_metrics": per_horizon_metrics,
        "elapsed_seconds": float(time.time() - t0),
    }

    with open(out_dir / f"metrics_prelen{cfg.pre_len}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    pred_denorm = y_pred_test * meta["scaler_y"]
    true_denorm = y_true_test * meta["scaler_y"]

    np.save(out_dir / f"pred_prelen{cfg.pre_len}.npy", pred_denorm)
    np.save(out_dir / f"true_prelen{cfg.pre_len}.npy", true_denorm)
    torch.save(model.state_dict(), out_dir / f"model_prelen{cfg.pre_len}.pt")

    print(f"\nSaved results to: {out_dir}")
    print(f"Elapsed: {time.time() - t0:.2f}s")


def main():
    args = build_arg_parser().parse_args()
    cfg = make_config(args)
    seed_all(cfg.random_state)

    for name in cfg.datasets:
        ds_cfg = DATASET_TABLE[name]
        run_one_dataset(ds_cfg, cfg)


if __name__ == "__main__":
    main()