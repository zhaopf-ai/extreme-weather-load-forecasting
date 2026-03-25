from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataprocess import WeatherImageLoader, load_data
from model import DCCALateFusionModel
from Trainer import Trainer
from helper import compute_maermse, compute_mape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--datasets",
        nargs="+",
        default=["Gympie"],
        choices=["Gympie", "Coolum", "Noosaville", "Tewantin"],
    )

    p.add_argument("--data_root", type=str, default=r"D:\python project\image extreme weather\data")
    p.add_argument("--exp_root", type=str, default="results")

    p.add_argument("--batch_size", type=int, default=32)
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

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--scheduler_factor", type=float, default=0.5)
    p.add_argument("--scheduler_patience", type=int, default=10)

    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--weather_feat_dim", type=int, default=4)

    p.add_argument("--image_seq_len", type=int, default=6)
    p.add_argument("--convlstm_hidden", type=int, default=64)
    p.add_argument("--convlstm_layers", type=int, default=1)
    p.add_argument("--convlstm_kernel", type=int, default=3)
    p.add_argument("--roi_size", type=int, default=8)
    p.add_argument("--pooled_size", type=int, default=4)
    p.add_argument("--ricnn_channels", type=int, default=64)

    p.add_argument("--gru_hidden", type=int, default=64)
    p.add_argument("--gru_layers", type=int, default=1)
    p.add_argument("--aux_hidden", type=int, default=64)
    p.add_argument("--fusion_hidden", type=int, default=64)
    p.add_argument("--pred_hidden", type=int, default=64)

    p.add_argument("--daily_len", type=int, default=6)
    p.add_argument("--weekly_len", type=int, default=4)

    p.add_argument("--dcca_dim", type=int, default=64)
    p.add_argument("--lambda_dcca", type=float, default=1e-5)
    p.add_argument("--cca_reg", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--seed", type=int, default=42)

    return p


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    img_subdir: str
    data_file: str
    processed_subdir: str


DATASET_TABLE: Dict[str, DatasetConfig] = {
    "Gympie": DatasetConfig(
        name="Gympie",
        img_subdir="modified",
        data_file="Gym.npz",
        processed_subdir="processed_image",
    ),
    "Coolum": DatasetConfig(
        name="Coolum",
        img_subdir="modified_other",
        data_file="Coo.npz",
        processed_subdir="processed_image_other",
    ),
    "Noosaville": DatasetConfig(
        name="Noosaville",
        img_subdir="modified_other",
        data_file="Noo.npz",
        processed_subdir="processed_image_other",
    ),
    "Tewantin": DatasetConfig(
        name="Tewantin",
        img_subdir="modified_other",
        data_file="Tew.npz",
        processed_subdir="processed_image_other",
    ),
}


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def evaluate(model, loader, scaler_y):
    model.eval()

    preds_all = []
    trues_all = []

    for weather, past, daily, weekly, time, targets, year, month, day, hour in loader:
        weather = weather.to(device)
        past = past.to(device)
        daily = daily.to(device)
        weekly = weekly.to(device)
        time = time.to(device)
        targets = targets.to(device)

        year = year.to(device)
        month = month.to(device)
        day = day.to(device)
        hour = hour.to(device)

        outputs = model(weather, past, daily, weekly, time, year, month, day, hour)

        if isinstance(outputs, tuple):
            pred = outputs[0]
        else:
            pred = outputs

        preds_all.append(pred.detach().cpu().numpy())
        trues_all.append(targets.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    trues_all = np.concatenate(trues_all, axis=0)

    rmse, mae = compute_maermse(preds_all, trues_all, scaler_y)
    mape = compute_mape(preds_all, trues_all, scaler_y)

    return preds_all, trues_all, rmse, mae, mape


def run_one_dataset(cfg: DatasetConfig, args: argparse.Namespace):
    data_root = Path(args.data_root)
    exp_root = Path(args.exp_root)
    exp_root.mkdir(parents=True, exist_ok=True)

    img_dir = data_root / cfg.img_subdir
    data_path = data_root / cfg.data_file
    processed_dir = data_root / cfg.processed_subdir

    run_tag = f"{cfg.name}_DCCA_LF"
    model_dir = exp_root / "model_saved"
    log_dir = exp_root / "logs"
    pred_dir = exp_root / "predictions"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"best_model_{run_tag}.pth"
    pred_path = pred_dir / f"{run_tag}_test_pred.npz"

    print(f"\n================ Dataset: {cfg.name} ================")
    print(f"img_dir      : {img_dir}")
    print(f"data_path    : {data_path}")
    print(f"processed_dir: {processed_dir}")
    print(f"model_path   : {model_path}")
    print("====================================================\n")

    weather_loader = WeatherImageLoader(str(img_dir), str(processed_dir))

    train_loader, val_loader, test_loader, scaler_y, time_feat_dim = load_data(
        str(data_path),
        batch_size=args.batch_size,
        his_length=args.his_len,
        pre_length=args.pre_len,
        add_weather_noise=args.add_weather_noise,
        noise_seed=args.noise_seed,
        train_end_date=args.train_end_date,
        val_start_date=args.val_start_date,
        val_end_date=args.val_end_date,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
    )

    model = DCCALateFusionModel(
        weather_loader=weather_loader,
        dim=args.dim,
        image_seq_len=args.image_seq_len,
        past_len=args.his_len,
        pred_len=args.pre_len,
        time_feat_dim=time_feat_dim,
        weather_feat_dim=args.weather_feat_dim,
        daily_len=args.daily_len,
        weekly_len=args.weekly_len,
        convlstm_hidden=args.convlstm_hidden,
        convlstm_layers=args.convlstm_layers,
        convlstm_kernel=args.convlstm_kernel,
        roi_size=args.roi_size,
        pooled_size=args.pooled_size,
        ricnn_channels=args.ricnn_channels,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        aux_hidden=args.aux_hidden,
        fusion_hidden=args.fusion_hidden,
        pred_hidden=args.pred_hidden,
        dcca_dim=args.dcca_dim,
        lambda_dcca=args.lambda_dcca,
        dcca_r1=args.cca_reg,
        dcca_r2=args.cca_reg,
        dropout=args.dropout,
    ).to(device)

    total_params, trainable_params = count_params(model)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler_y=scaler_y,
        epochs=args.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        save_dir=str(model_path),
    )
    trainer.train()

    model.load_state_dict(torch.load(model_path, map_location=device))

    preds, trues, rmse, mae, mape = evaluate(model, test_loader, scaler_y)

    np.savez(
        pred_path,
        predictions=preds,
        actuals=trues,
    )

    print(f"[{cfg.name}] Test RMSE: {rmse:.4f}")
    print(f"[{cfg.name}] Test MAE : {mae:.4f}")
    print(f"[{cfg.name}] Test MAPE: {mape:.2f}%")
    print(f"[{cfg.name}] Predictions saved to: {pred_path}")


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    for dataset_name in args.datasets:
        cfg = DATASET_TABLE[dataset_name]
        run_one_dataset(cfg, args)


if __name__ == "__main__":
    main()