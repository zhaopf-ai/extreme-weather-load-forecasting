from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataprocess import WeatherImageLoader, load_data
from Invimt import InViMT
from Trainer import Trainer
from helper import compute_maermse, compute_mape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--datasets",
        nargs="+",
        default=["Gympie"],
        choices=["Gympie", "Coolum", "Noosaville", "Tewantin"],
        help="Which datasets to run",
    )

    p.add_argument(
        "--data_root",
        type=str,
        default=r"D:\python project\image extreme weather\data",
    )
    p.add_argument(
        "--exp_root",
        type=str,
        default=r"D:\python project\image_extreme_weather_v2\results",
    )

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--his_len", type=int, default=48)
    p.add_argument("--pre_len", type=int, default=1)

    p.add_argument("--add_weather_noise", action="store_true", default=True)
    p.add_argument("--no_weather_noise", dest="add_weather_noise", action="store_false")
    p.add_argument("--noise_seed", type=int, default=42)

    p.add_argument("--train_end_date", type=str, default="2021-11-30")
    p.add_argument("--val_start_date", type=str, default="2021-12-01")
    p.add_argument("--val_end_date", type=str, default="2022-02-10")
    p.add_argument("--test_start_date", type=str, default="2022-02-11")
    p.add_argument("--test_end_date", type=str, default="2022-03-04")

    p.add_argument("--image_seq_len", type=int, default=6)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)

    p.add_argument("--weather_feat_dim", type=int, default=4)

    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--informer_layers", type=int, default=2)
    p.add_argument("--vit_layers", type=int, default=4)
    p.add_argument("--decoder_layers", type=int, default=1)
    p.add_argument("--sparse_factor", type=float, default=0.6)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--scheduler_factor", type=float, default=0.5)
    p.add_argument("--scheduler_patience", type=int, default=10)

    p.add_argument("--loss", type=str, default="l1", choices=["l1", "mse"])
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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_criterion(loss_name: str) -> nn.Module:
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def evaluate_model(model: nn.Module, test_loader, scaler_y: float) -> None:
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for weather, past, daily, weekly, time, targets, year, month, day, hour in test_loader:
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

            outputs = model(
                weather,
                past,
                daily,
                weekly,
                time,
                year,
                month,
                day,
                hour,
            )

            preds.append(outputs.detach().cpu().numpy())
            trues.append(targets.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    preds_denorm = preds * scaler_y
    trues_denorm = trues * scaler_y

    mape = compute_mape(trues_denorm, preds_denorm)
    mae, rmse = compute_maermse(trues_denorm, preds_denorm)

    print("\n================ Test Results ================")
    print(f"MAPE: {mape:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("=============================================\n")


def run_one_dataset(cfg: DatasetConfig, args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    exp_root = Path(args.exp_root)
    exp_root.mkdir(parents=True, exist_ok=True)

    img_dir = data_root / cfg.img_subdir
    data_path = data_root / cfg.data_file
    processed_dir = data_root / cfg.processed_subdir

    run_tag = f"{cfg.name}_InViMT"
    model_path = exp_root / "model_saved" / f"best_model_{run_tag}.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n================ Dataset: {cfg.name} ================")
    print(f"img_dir      : {img_dir}")
    print(f"data_path    : {data_path}")
    print(f"processed_dir: {processed_dir}")
    print(f"model_path   : {model_path}")
    print("=====================================================\n")

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

    model = InViMT(
        weather_loader=weather_loader,
        his_len=args.his_len,
        pred_len=args.pre_len,
        image_seq_len=args.image_seq_len,
        weather_feat_dim=args.weather_feat_dim,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        informer_layers=args.informer_layers,
        vit_layers=args.vit_layers,
        decoder_layers=args.decoder_layers,
        sparse_factor=args.sparse_factor,
        patch_size=args.patch_size,
        image_size=args.image_size,
        dropout=args.dropout,
    ).to(device)

    total_params, trainable_params = count_params(model)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    criterion = build_criterion(args.loss)

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
        verbose=True,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler_y=scaler_y,
        epochs=args.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        scheduler=scheduler,
        save_dir=str(model_path),
        run_tag=run_tag,
        log_dir=str(exp_root / "logs"),
    )
    trainer.train()

    best_model = InViMT(
        weather_loader=weather_loader,
        his_len=args.his_len,
        pred_len=args.pre_len,
        image_seq_len=args.image_seq_len,
        weather_feat_dim=args.weather_feat_dim,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        informer_layers=args.informer_layers,
        vit_layers=args.vit_layers,
        decoder_layers=args.decoder_layers,
        sparse_factor=args.sparse_factor,
        patch_size=args.patch_size,
        image_size=args.image_size,
        dropout=args.dropout,
    ).to(device)

    best_model.load_state_dict(torch.load(model_path, map_location=device))
    evaluate_model(best_model, test_loader, scaler_y)


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    for dataset_name in args.datasets:
        cfg = DATASET_TABLE[dataset_name]
        run_one_dataset(cfg, args)


if __name__ == "__main__":
    main()