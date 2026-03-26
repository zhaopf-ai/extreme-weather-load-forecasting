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
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer


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

    n_knots: int = 8
    degree: int = 3
    ridge_alpha: float = 1.0

    random_state: int = 42


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

    p.add_argument("--n_knots", type=int, default=8)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--ridge_alpha", type=float, default=1.0)

    p.add_argument("--random_state", type=int, default=42)

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
        n_knots=args.n_knots,
        degree=args.degree,
        ridge_alpha=args.ridge_alpha,
        random_state=args.random_state,
    )


def build_gam_model(cfg: TrainConfig) -> Pipeline:
    return Pipeline([
        (
            "spline",
            SplineTransformer(
                n_knots=cfg.n_knots,
                degree=cfg.degree,
                include_bias=False,
            ),
        ),
        (
            "ridge",
            Ridge(alpha=cfg.ridge_alpha, random_state=cfg.random_state),
        ),
    ])


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
    X_lag = []
    X_weather = []
    X_time = []
    Y = []
    ts_start = []

    for i in range(his_length, n - pre_length + 1):
        x_past = load_norm[i - his_length:i]

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

        X_past.append(x_past.astype(np.float32))
        X_lag.append(x_lag.astype(np.float32))
        X_weather.append(x_weather.astype(np.float32))
        X_time.append(x_time.astype(np.float32))
        Y.append(y.astype(np.float32))
        ts_start.append(row_ts.iloc[i])

    X_past = np.asarray(X_past, dtype=np.float32)
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
        "past_feat_dim": int(X_past.shape[1]),
        "lag_feat_dim": int(X_lag.shape[2]),
        "weather_feat_dim": int(X_weather.shape[2]),
        "time_feat_dim": int(X_time.shape[2]),
        "feature_dim_per_horizon": int(X_past.shape[1] + X_lag.shape[2] + X_weather.shape[2] + X_time.shape[2]),
    }

    return X_past, X_lag, X_weather, X_time, Y, split, meta


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

    print(f"\n================ Dataset: {ds_cfg.name} ================")
    print(f"data_path : {data_path}")
    print("========================================================")

    t0 = time.time()

    X_past, X_lag, X_weather, X_time, y, split, meta = build_samples_from_file(
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

    X_past_train, X_lag_train, X_weather_train, X_time_train, y_train = (
        X_past[train_mask], X_lag[train_mask], X_weather[train_mask], X_time[train_mask], y[train_mask]
    )
    X_past_val, X_lag_val, X_weather_val, X_time_val, y_val = (
        X_past[val_mask], X_lag[val_mask], X_weather[val_mask], X_time[val_mask], y[val_mask]
    )
    X_past_test, X_lag_test, X_weather_test, X_time_test, y_test = (
        X_past[test_mask], X_lag[test_mask], X_weather[test_mask], X_time[test_mask], y[test_mask]
    )

    if len(X_past_train) == 0 or len(X_past_val) == 0 or len(X_past_test) == 0:
        raise ValueError(
            f"Empty split detected: train={len(X_past_train)}, val={len(X_past_val)}, test={len(X_past_test)}"
        )

    X_past_trv = np.concatenate([X_past_train, X_past_val], axis=0)
    X_lag_trv = np.concatenate([X_lag_train, X_lag_val], axis=0)
    X_weather_trv = np.concatenate([X_weather_train, X_weather_val], axis=0)
    X_time_trv = np.concatenate([X_time_train, X_time_val], axis=0)
    y_trv = np.concatenate([y_train, y_val], axis=0)

    print(f"Feature dim per horizon: {meta['feature_dim_per_horizon']}")
    print(f"Train: {len(X_past_train)} | Val: {len(X_past_val)} | Test: {len(X_past_test)}")
    print(f"Train range: <= {cfg.train_end_date}")
    print(f"Val range  : {cfg.val_start_date} to {cfg.val_end_date}")
    print(f"Test range : {cfg.test_start_date} to {cfg.test_end_date}")

    preds_test = []
    models = []

    for h in range(cfg.pre_len):
        Xh_trv = np.concatenate(
            [
                X_past_trv,
                X_lag_trv[:, h, :],
                X_weather_trv[:, h, :],
                X_time_trv[:, h, :],
            ],
            axis=1,
        )
        Xh_test = np.concatenate(
            [
                X_past_test,
                X_lag_test[:, h, :],
                X_weather_test[:, h, :],
                X_time_test[:, h, :],
            ],
            axis=1,
        )

        model = build_gam_model(cfg)
        model.fit(Xh_trv, y_trv[:, h])
        pred_h = model.predict(Xh_test).reshape(-1, 1)

        preds_test.append(pred_h)
        models.append(model)
        print(f"Horizon {h + 1}: finished, input shape = {Xh_trv.shape}")

    y_pred_test = np.concatenate(preds_test, axis=1)

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
        "num_train": int(len(X_past_train)),
        "num_val": int(len(X_past_val)),
        "num_test": int(len(X_past_test)),
        "overall_metrics": overall_metrics,
        "per_horizon_metrics": per_horizon_metrics,
        "elapsed_seconds": float(time.time() - t0),
    }

    with open(out_dir / f"metrics_prelen{cfg.pre_len}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    pred_denorm = y_pred_test * meta["scaler_y"]
    true_denorm = y_test * meta["scaler_y"]

    np.save(out_dir / f"pred_prelen{cfg.pre_len}.npy", pred_denorm)
    np.save(out_dir / f"true_prelen{cfg.pre_len}.npy", true_denorm)

    print(f"\nSaved results to: {out_dir}")
    print(f"Elapsed: {time.time() - t0:.2f}s")


def main():
    args = build_arg_parser().parse_args()
    cfg = make_config(args)

    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    for name in cfg.datasets:
        ds_cfg = DATASET_TABLE[name]
        run_one_dataset(ds_cfg, cfg)


if __name__ == "__main__":
    main()
