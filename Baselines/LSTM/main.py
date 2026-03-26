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
    pre_len: int = 1

    add_weather_noise: bool = True
    noise_seed: int = 42

    train_end_date: str = "2021-11-30"
    val_start_date: str = "2021-12-01"
    val_end_date: str = "2022-02-10"
    test_start_date: str = "2022-02-11"
    test_end_date: str = "2022-03-04"

    lstm_layers: int = 2
    lstm_units: int = 100
    hist_dense_units: int = 32

    lag_mlp_units1: int = 64
    lag_mlp_units2: int = 32
    weather_mlp_units1: int = 64
    weather_mlp_units2: int = 32
    time_mlp_units1: int = 64
    time_mlp_units2: int = 32

    fusion_units1: int = 128
    fusion_units2: int = 64

    lr: float = 1e-3
    lr_factor: float = 0.5
    lr_patience: int = 10
    min_lr: float = 1e-6

    batch_size: int = 128
    epochs: int = 5
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
    p.add_argument("--save_root", type=str, default="results")

    p.add_argument("--his_len", type=int, default=6)
    p.add_argument("--pre_len", type=int, default=1)

    p.add_argument("--add_weather_noise", action="store_true", default=True)
    p.add_argument("--no_weather_noise", dest="add_weather_noise", action="store_false")
    p.add_argument("--noise_seed", type=int, default=42)

    p.add_argument("--train_end_date", type=str, default="2021-11-30")
    p.add_argument("--val_start_date", type=str, default="2021-12-01")
    p.add_argument("--val_end_date", type=str, default="2022-02-10")
    p.add_argument("--test_start_date", type=str, default="2022-02-11")
    p.add_argument("--test_end_date", type=str, default="2022-03-04")

    p.add_argument("--lstm_layers", type=int, default=2)
    p.add_argument("--lstm_units", type=int, default=100)
    p.add_argument("--hist_dense_units", type=int, default=32)

    p.add_argument("--lag_mlp_units1", type=int, default=64)
    p.add_argument("--lag_mlp_units2", type=int, default=32)
    p.add_argument("--weather_mlp_units1", type=int, default=64)
    p.add_argument("--weather_mlp_units2", type=int, default=32)
    p.add_argument("--time_mlp_units1", type=int, default=64)
    p.add_argument("--time_mlp_units2", type=int, default=32)

    p.add_argument("--fusion_units1", type=int, default=128)
    p.add_argument("--fusion_units2", type=int, default=64)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_factor", type=float, default=0.99)
    p.add_argument("--lr_patience", type=int, default=10)
    p.add_argument("--min_lr", type=float, default=1e-6)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
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
        lstm_layers=args.lstm_layers,
        lstm_units=args.lstm_units,
        hist_dense_units=args.hist_dense_units,
        lag_mlp_units1=args.lag_mlp_units1,
        lag_mlp_units2=args.lag_mlp_units2,
        weather_mlp_units1=args.weather_mlp_units1,
        weather_mlp_units2=args.weather_mlp_units2,
        time_mlp_units1=args.time_mlp_units1,
        time_mlp_units2=args.time_mlp_units2,
        fusion_units1=args.fusion_units1,
        fusion_units2=args.fusion_units2,
        lr=args.lr,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        min_lr=args.min_lr,
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

    X_hist = []
    X_lag = []
    X_weather = []
    X_time = []
    Y = []
    ts_start = []

    for i in range(his_length, n - pre_length + 1):
        x_hist = load_norm[i - his_length:i].reshape(his_length, 1)

        x_daily = daily[i:i + pre_length]
        x_weekly = weekly[i:i + pre_length]
        x_lag = np.concatenate([x_daily, x_weekly], axis=1)

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

        X_hist.append(x_hist)
        X_lag.append(x_lag.astype(np.float32))
        X_weather.append(x_weather.astype(np.float32))
        X_time.append(x_time.astype(np.float32))
        Y.append(y.astype(np.float32))
        ts_start.append(row_ts.iloc[i])

    X_hist = np.asarray(X_hist, dtype=np.float32)
    X_lag = np.asarray(X_lag, dtype=np.float32)
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
        "history_len": int(his_length),
        "hist_dim": 1,
        "lag_dim": int(X_lag.shape[2]),
        "weather_dim": int(X_weather.shape[2]),
        "time_dim": int(X_time.shape[2]),
        "pre_len": int(pre_length),
    }

    return X_hist, X_lag, X_weather, X_time, Y, split, meta


class LSTM(nn.Module):
    def __init__(
        self,
        hist_dim: int,
        lag_dim: int,
        weather_dim: int,
        time_dim: int,
        pre_len: int,
        lstm_layers: int = 2,
        lstm_units: int = 100,
        hist_dense_units: int = 32,
        lag_mlp_units1: int = 64,
        lag_mlp_units2: int = 32,
        weather_mlp_units1: int = 64,
        weather_mlp_units2: int = 32,
        time_mlp_units1: int = 64,
        time_mlp_units2: int = 32,
        fusion_units1: int = 128,
        fusion_units2: int = 64,
    ):
        super().__init__()
        self.pre_len = pre_len

        self.lstm = nn.LSTM(
            input_size=hist_dim,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.hist_dense = nn.Linear(lstm_units, hist_dense_units)

        self.lag_mlp1 = nn.Linear(lag_dim, lag_mlp_units1)
        self.lag_mlp2 = nn.Linear(lag_mlp_units1, lag_mlp_units2)

        self.weather_mlp1 = nn.Linear(weather_dim, weather_mlp_units1)
        self.weather_mlp2 = nn.Linear(weather_mlp_units1, weather_mlp_units2)

        self.time_mlp1 = nn.Linear(time_dim, time_mlp_units1)
        self.time_mlp2 = nn.Linear(time_mlp_units1, time_mlp_units2)

        fusion_in_dim = hist_dense_units + lag_mlp_units2 + weather_mlp_units2 + time_mlp_units2
        self.fusion_mlp1 = nn.Linear(fusion_in_dim, fusion_units1)
        self.fusion_mlp2 = nn.Linear(fusion_units1, fusion_units2)
        self.out = nn.Linear(fusion_units2, 1)

    def forward(self, x_hist, x_lag, x_weather, x_time):
        h, _ = self.lstm(x_hist)
        h = h[:, -1, :]
        h = F.relu(self.hist_dense(h))
        h = h.unsqueeze(1).expand(-1, self.pre_len, -1)

        lag_feat = F.relu(self.lag_mlp1(x_lag))
        lag_feat = F.relu(self.lag_mlp2(lag_feat))

        weather_feat = F.relu(self.weather_mlp1(x_weather))
        weather_feat = F.relu(self.weather_mlp2(weather_feat))

        time_feat = F.relu(self.time_mlp1(x_time))
        time_feat = F.relu(self.time_mlp2(time_feat))

        z = torch.cat([h, lag_feat, weather_feat, time_feat], dim=-1)
        z = F.relu(self.fusion_mlp1(z))
        z = F.relu(self.fusion_mlp2(z))
        y = self.out(z).squeeze(-1)
        return y


def make_loader(X_hist, X_lag, X_weather, X_time, Y, mask, batch_size, shuffle):
    xh = torch.from_numpy(X_hist[mask]).float()
    xl = torch.from_numpy(X_lag[mask]).float()
    xw = torch.from_numpy(X_weather[mask]).float()
    xt = torch.from_numpy(X_time[mask]).float()
    y = torch.from_numpy(Y[mask]).float()

    ds = TensorDataset(xh, xl, xw, xt, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate_model(model, loader, device):
    model.eval()
    preds = []
    trues = []
    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for xh, xl, xw, xt, y in loader:
            xh = xh.to(device)
            xl = xl.to(device)
            xw = xw.to(device)
            xt = xt.to(device)
            y = y.to(device)

            pred = model(xh, xl, xw, xt)
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
        min_lr=cfg.min_lr,
    )

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    lr_history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xh, xl, xw, xt, y in train_loader:
            xh = xh.to(device)
            xl = xl.to(device)
            xw = xw.to(device)
            xt = xt.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xh, xl, xw, xt)
            loss = mape_loss(pred, y)
            loss.backward()
            optimizer.step()

        val_loss, _, _ = evaluate_model(model, val_loader, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(float(current_lr))

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
    return model, best_val, best_epoch, lr_history


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

    X_hist, X_lag, X_weather, X_time, Y, split, meta = build_samples_from_file(
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

    train_loader = make_loader(X_hist, X_lag, X_weather, X_time, Y, train_mask, cfg.batch_size, False)
    val_loader = make_loader(X_hist, X_lag, X_weather, X_time, Y, val_mask, cfg.batch_size, False)
    test_loader = make_loader(X_hist, X_lag, X_weather, X_time, Y, test_mask, cfg.batch_size, False)

    model = LSTM(
        hist_dim=meta["hist_dim"],
        lag_dim=meta["lag_dim"],
        weather_dim=meta["weather_dim"],
        time_dim=meta["time_dim"],
        pre_len=cfg.pre_len,
        lstm_layers=cfg.lstm_layers,
        lstm_units=cfg.lstm_units,
        hist_dense_units=cfg.hist_dense_units,
        lag_mlp_units1=cfg.lag_mlp_units1,
        lag_mlp_units2=cfg.lag_mlp_units2,
        weather_mlp_units1=cfg.weather_mlp_units1,
        weather_mlp_units2=cfg.weather_mlp_units2,
        time_mlp_units1=cfg.time_mlp_units1,
        time_mlp_units2=cfg.time_mlp_units2,
        fusion_units1=cfg.fusion_units1,
        fusion_units2=cfg.fusion_units2,
    ).to(device)

    print(f"Train: {int(train_mask.sum())} | Val: {int(val_mask.sum())} | Test: {int(test_mask.sum())}")
    print(f"Train range: <= {cfg.train_end_date}")
    print(f"Val range  : {cfg.val_start_date} to {cfg.val_end_date}")
    print(f"Test range : {cfg.test_start_date} to {cfg.test_end_date}")
    print(f"train_hist={X_hist[train_mask].shape}")
    print(f"train_lag={X_lag[train_mask].shape}")
    print(f"train_weather={X_weather[train_mask].shape}")
    print(f"train_time={X_time[train_mask].shape}")
    print(f"train_y={Y[train_mask].shape}")

    model, best_val, best_epoch, lr_history = train_model(model, train_loader, val_loader, cfg, device)

    _, y_pred_test, y_true_test = evaluate_model(model, test_loader, device)

    overall_metrics, per_horizon_metrics = evaluate_multistep(
        y_true_norm=y_true_test,
        y_pred_norm=y_pred_test,
        scaler_y=meta["scaler_y"],
    )

    print(f"\nBest val MAPE: {best_val:.6f} @ epoch {best_epoch}")
    print(f"Final learning rate: {lr_history[-1]:.8f}" if len(lr_history) > 0 else "\nFinal learning rate: N/A")
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
        "lr_history": lr_history,
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
