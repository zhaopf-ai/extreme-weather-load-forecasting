from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import gpytorch
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

    hidden_dim: int = 20
    num_res_blocks: int = 3
    nn_lr: float = 1e-3
    nn_epochs: int = 100
    batch_size: int = 128

    gp_lengthscale: float = 0.1
    gp_lr: float = 1e-2
    gp_epochs: int = 800
    num_inducing: int = 200

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

    p.add_argument("--hidden_dim", type=int, default=20)
    p.add_argument("--num_res_blocks", type=int, default=3)
    p.add_argument("--nn_lr", type=float, default=1e-3)
    p.add_argument("--nn_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)

    p.add_argument("--gp_lengthscale", type=float, default=0.1)
    p.add_argument("--gp_lr", type=float, default=1e-2)
    p.add_argument("--gp_epochs", type=int, default=800)
    p.add_argument("--num_inducing", type=int, default=200)

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
        nn_lr=args.nn_lr,
        nn_epochs=args.nn_epochs,
        batch_size=args.batch_size,
        gp_lengthscale=args.gp_lengthscale,
        gp_lr=args.gp_lr,
        gp_epochs=args.gp_epochs,
        num_inducing=args.num_inducing,
        random_state=args.random_state,
        device=args.device,
    )


class ResDenseBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.proj = None if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim)

        nn.init.normal_(self.fc1.weight, 0.0, np.sqrt(1.0 / in_dim))
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, 0.0, np.sqrt(1.0 / hidden_dim))
        nn.init.zeros_(self.fc2.bias)
        if self.proj is not None:
            nn.init.normal_(self.proj.weight, 0.0, np.sqrt(1.0 / in_dim))
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        h = F.selu(self.fc1(x))
        h = self.fc2(h)
        s = x if self.proj is None else self.proj(x)
        return s + h


class ResMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=20, num_res_blocks=3):
        super().__init__()
        self.block0 = ResDenseBlock(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResDenseBlock(hidden_dim, hidden_dim) for _ in range(num_res_blocks)])
        self.in_skip = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

        nn.init.normal_(self.in_skip.weight, 0.0, np.sqrt(1.0 / in_dim))
        nn.init.zeros_(self.in_skip.bias)
        nn.init.normal_(self.out.weight, 0.0, np.sqrt(1.0 / hidden_dim))
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = self.block0(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h + self.in_skip(x))


class VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, input_dimension, output_dimension, lengthscale):
        q = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        s = gpytorch.variational.VariationalStrategy(
            self, inducing_points, q, learn_inducing_locations=True
        )
        super().__init__(s)

        self.mean_module = gpytorch.means.ZeroMean()

        self.k1 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(active_dims=tuple(range(input_dimension)))
        )
        self.k2 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                active_dims=tuple(range(input_dimension, input_dimension + output_dimension))
            )
        )

        self.k1.base_kernel.lengthscale = lengthscale
        self.k2.base_kernel.lengthscale = lengthscale

        self.covar_module = self.k1 + self.k2

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
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

    feats = []
    targets = []
    ts_start = []

    for i in range(his_length, n - pre_length + 1):
        x_past = load_norm[i - his_length:i]

        sample_x = []
        sample_y = []

        for h in range(pre_length):
            idx = i + h
            lead = h + 1

            x_daily = daily[idx]
            x_weekly = weekly[idx]

            if add_weather_noise:
                w = weather_raw[idx].copy()
                w[0] += rng.normal(0, _interp_sigma(lead, 0.5, 2.0))
                w[1] += rng.normal(0, _interp_sigma(lead, 1.0, 2.0) * 3.6)
                w[3] += rng.normal(0, _interp_sigma(lead, 5.0, 12.0))
                w[2] *= 1.0 + rng.normal(0, precip_cv_1to9[min(lead, 9) - 1])
                w[1] = np.clip(w[1], 0, None)
                w[2] = np.clip(w[2], 0, None)
                w[3] = np.clip(w[3], 0, 100)
                x_weather = w / weather_max
            else:
                x_weather = weather[idx]

            x_time = time[idx]
            y_h = load_norm[idx]

            x_h = np.concatenate([x_past, x_daily, x_weekly, x_weather, x_time], axis=0)
            sample_x.append(x_h)
            sample_y.append(y_h)

        feats.append(np.stack(sample_x, axis=0))
        targets.append(np.asarray(sample_y, dtype=np.float32))
        ts_start.append(row_ts.iloc[i])

    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)

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
        "feature_dim": int(X.shape[2]),
        "past_feat_dim": int(his_length),
        "daily_feat_dim": int(len(daily_cols)),
        "weekly_feat_dim": int(len(weekly_cols)),
        "weather_feat_dim": int(len(weather_cols)),
        "time_feat_dim": int(len(time_cols)),
    }

    return X, y, split, meta


def train_backbone(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    cfg: TrainConfig,
    device,
):
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float().reshape(-1, 1),
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().reshape(-1, 1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.nn_lr)

    best_state = None
    best_val = float("inf")

    for ep in range(1, cfg.nn_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = mape_loss(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mape = mape_loss(model(X_val_t), y_val_t).item()

        if val_mape < best_val:
            best_val = val_mape
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val


def fit_gp_residual(
    backbone,
    X_gp,
    y_gp,
    X_test,
    cfg: TrainConfig,
    device,
):
    backbone.eval()

    X_gp_t = torch.from_numpy(X_gp).float().to(device)
    y_gp_t = torch.from_numpy(y_gp).float().reshape(-1, 1).to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)

    with torch.no_grad():
        train_pred = backbone(X_gp_t).detach()
        test_pred = backbone(X_test_t).detach()

    residual = (y_gp_t - train_pred).reshape(-1, 1)

    gp_feat_train = torch.cat([X_gp_t, train_pred], dim=1)
    gp_feat_test = torch.cat([X_test_t, test_pred], dim=1)

    m = min(cfg.num_inducing, gp_feat_train.size(0))
    Z = gp_feat_train[:m, :].clone()

    input_dimension = X_gp.shape[1]
    output_dimension = 1

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    gp_model = VariationalGPModel(
        inducing_points=Z,
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        lengthscale=cfg.gp_lengthscale,
    ).to(device)

    gp_model.train()
    likelihood.train()

    mll = gpytorch.mlls.VariationalELBO(likelihood, gp_model, num_data=gp_feat_train.size(0))
    optimizer = torch.optim.Adam(
        [{"params": gp_model.parameters()}, {"params": likelihood.parameters()}],
        lr=cfg.gp_lr,
    )

    y_gp_vec = residual.reshape(-1)

    for _ in range(cfg.gp_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = -mll(gp_model(gp_feat_train), y_gp_vec)
        loss.backward()
        optimizer.step()

    gp_model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(gp_model(gp_feat_test))
        gp_mean = pred_dist.mean.detach().cpu().numpy().reshape(-1)
        gp_var = pred_dist.variance.detach().cpu().numpy().reshape(-1)

    nn_pred = test_pred.detach().cpu().numpy().reshape(-1)
    final_pred = nn_pred + gp_mean

    return final_pred, gp_var


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

    X, y, split, meta = build_samples_from_file(
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

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Empty split detected: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )

    print(f"Feature dim: {meta['feature_dim']}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Train range: <= {cfg.train_end_date}")
    print(f"Val range  : {cfg.val_start_date} to {cfg.val_end_date}")
    print(f"Test range : {cfg.test_start_date} to {cfg.test_end_date}")

    preds_test = []
    vars_test = []
    best_val_mapes = []

    for h in range(cfg.pre_len):
        X_train_h = X_train[:, h, :]
        y_train_h = y_train[:, h]
        X_val_h = X_val[:, h, :]
        y_val_h = y_val[:, h]
        X_test_h = X_test[:, h, :]

        X_gp_h = np.concatenate([X_train_h, X_val_h], axis=0)
        y_gp_h = np.concatenate([y_train_h, y_val_h], axis=0)

        backbone = ResMLP(
            in_dim=X.shape[2],
            hidden_dim=cfg.hidden_dim,
            num_res_blocks=cfg.num_res_blocks,
        ).to(device)

        backbone, best_val = train_backbone(
            model=backbone,
            X_train=X_train_h,
            y_train=y_train_h,
            X_val=X_val_h,
            y_val=y_val_h,
            cfg=cfg,
            device=device,
        )

        pred_h, var_h = fit_gp_residual(
            backbone=backbone,
            X_gp=X_gp_h,
            y_gp=y_gp_h,
            X_test=X_test_h,
            cfg=cfg,
            device=device,
        )

        preds_test.append(pred_h.reshape(-1, 1))
        vars_test.append(var_h.reshape(-1, 1))
        best_val_mapes.append(best_val)

        print(f"Horizon {h + 1}: best_val_mape={best_val:.6f}")

    y_pred_test = np.concatenate(preds_test, axis=1)
    y_var_test = np.concatenate(vars_test, axis=1)

    overall_metrics, per_horizon_metrics = evaluate_multistep(
        y_true_norm=y_test,
        y_pred_norm=y_pred_test,
        scaler_y=meta["scaler_y"],
    )

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
        "num_train": int(len(X_train)),
        "num_val": int(len(X_val)),
        "num_test": int(len(X_test)),
        "best_val_mapes": best_val_mapes,
        "overall_metrics": overall_metrics,
        "per_horizon_metrics": per_horizon_metrics,
        "elapsed_seconds": float(time.time() - t0),
    }

    with open(out_dir / f"metrics_prelen{cfg.pre_len}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    pred_denorm = y_pred_test * meta["scaler_y"]
    true_denorm = y_test * meta["scaler_y"]
    var_denorm = y_var_test * (meta["scaler_y"] ** 2)

    np.save(out_dir / f"pred_prelen{cfg.pre_len}.npy", pred_denorm)
    np.save(out_dir / f"true_prelen{cfg.pre_len}.npy", true_denorm)
    np.save(out_dir / f"var_prelen{cfg.pre_len}.npy", var_denorm)

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
