import os
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _read_tabular(file_path: str) -> pd.DataFrame:
    """Read tabular data from .csv or .npz.

    - .csv: read via pandas
    - .npz: expects keys {data, columns}
    """
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

        raise ValueError(
            f"Invalid npz format: {fp}. Expected 'records' or ('data' and 'columns')."
        )

    raise ValueError(f"Unsupported file type: {fp}. Use .csv or .npz")


def load_data_from_file(
    file_path,
    batch_size=32,
    his_length=24,
    pre_length=6,
    add_weather_noise=True,
    noise_seed=42,
    train_end_date="2021-11-30",
    val_start_date="2021-12-01",
    val_end_date="2021-12-31",
    test_start_date="2022-01-01",
    test_end_date=None,
):
    # Supports both .csv and .npz (recommended).
    data = _read_tabular(file_path)

    year = data["year"].values.astype(np.int32)
    month = data["month"].values.astype(np.int32)
    day = data["day"].values.astype(np.int32)
    hour = data["hour"].values.astype(np.int32)

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

    weather_raw = data[weather_cols].values.astype(np.float32)
    time_raw = data[time_cols].values.astype(np.float32)

    rng = np.random.default_rng(noise_seed)
    precip_cv_1to9 = np.array([0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.25, 1.30], dtype=np.float32)

    def _interp_sigma(lead, s1, s9):
        lead = int(np.clip(lead, 1, 9))
        return s1 + (lead - 1) * (s9 - s1) / 8.0

    weather_max = np.maximum(weather_raw[train_row_mask].max(axis=0), 1e-6)
    weather = weather_raw / weather_max

    time_max = np.maximum(time_raw[train_row_mask].max(axis=0), 1e-6)
    time = time_raw / time_max
    time_feat_dim = time.shape[1]

    N = len(load_norm)
    if N < his_length + pre_length:
        raise ValueError("Insufficient data length.")

    X_past, y, X_w, X_t = [], [], [], []
    Y, M, D, H = [], [], [], []
    ts_start = []

    for i in range(his_length, N - pre_length + 1):
        X_past.append(load_norm[i - his_length:i])
        y.append(load_norm[i:i + pre_length])

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
            w = w / weather_max
        else:
            w = weather[i:i + pre_length]

        t = time[i:i + pre_length]

        X_w.append(w.reshape(-1))
        X_t.append(t.reshape(-1))

        Y.append(year[i])
        M.append(month[i])
        D.append(day[i])
        H.append(hour[i])
        ts_start.append(row_ts.iloc[i])

    X_past = np.asarray(X_past, np.float32)
    y = np.asarray(y, np.float32)
    X_w = np.asarray(X_w, np.float32)
    X_t = np.asarray(X_t, np.float32)

    Y = np.asarray(Y, np.float32).reshape(-1, 1)
    M = np.asarray(M, np.float32).reshape(-1, 1)
    D = np.asarray(D, np.float32).reshape(-1, 1)
    H = np.asarray(H, np.float32).reshape(-1, 1)

    ts_start = pd.to_datetime(ts_start)
    ts_end = ts_start + pd.to_timedelta(pre_length - 1, unit="h")

    train_mask = ts_end <= train_end_ts
    val_mask = (ts_start >= val_start_ts) & (ts_end <= val_end_ts)
    test_mask = ts_start >= test_start_ts if test_end_ts is None else (
        (ts_start >= test_start_ts) & (ts_end <= test_end_ts)
    )

    def _build_loader(mask):
        return TensorDataset(
            torch.tensor(X_w[mask]),
            torch.tensor(X_past[mask]),
            torch.tensor(X_t[mask]),
            torch.tensor(y[mask]),
            torch.tensor(Y[mask]),
            torch.tensor(M[mask]),
            torch.tensor(D[mask]),
            torch.tensor(H[mask]),
        )

    train_loader = DataLoader(_build_loader(train_mask), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(_build_loader(val_mask), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(_build_loader(test_mask), batch_size=batch_size, shuffle=False)

    print(f"[Split] train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    return train_loader, val_loader, test_loader, scaler_y, time_feat_dim




def load_data(*args, **kwargs):
    return load_data_from_file(*args, **kwargs)


class WeatherImageLoader:

    def __init__(self, img_dir, processed_dir, image_size=(224, 224)):
        self.img_dir = img_dir
        self.processed_dir = processed_dir
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4554, 0.4645, 0.3512],
                std=[0.1292, 0.1194, 0.1167],
            ),
        ])

        os.makedirs(self.processed_dir, exist_ok=True)

    def load_images_for_hour(self, timestamp):
        """Return cached image tensor or build from 15-min intervals."""
        cache_name = f"cached_{timestamp.strftime('%Y%m%dT%H%M%S')}.pt"
        cache_path = os.path.join(self.processed_dir, cache_name)

        if os.path.exists(cache_path):
            return torch.load(cache_path)

        images = []
        for i in range(4):
            ts = timestamp + timedelta(minutes=i * 15)
            name = f"Satellite-With-Radar_{ts.strftime('%Y%m%dT%H%M%S')}+0800_modified.png"
            path = os.path.join(self.img_dir, name)

            if not os.path.exists(path):
                continue

            try:
                img = Image.open(path).convert("RGB")
                images.append(self.transform(img))
            except (OSError, IOError, UnidentifiedImageError):
                continue

        if images:
            out = torch.mean(torch.stack(images), dim=0)
        else:
            out = torch.zeros((3, *self.image_size), dtype=torch.float32)

        torch.save(out, cache_path)
        return out
