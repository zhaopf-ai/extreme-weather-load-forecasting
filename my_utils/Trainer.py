import os
import time as T
from datetime import datetime
import numpy as np
import torch
from my_utils.mixup import c_mixup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync_cuda():
    """Synchronize CUDA for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scaler_y,
        epochs,
        train_loader,
        val_loader=None,
        test_loader=None,
        scheduler=None,
        save_dir=None,
        run_tag: str = "",
        log_dir: str = "logs",
        extra_log_header: str = "",   
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler_y = scaler_y
        self.epochs = epochs
        self.extra_log_header = extra_log_header

        self.train_loader = train_loader
        self.val_loader = val_loader if val_loader is not None else test_loader
        self.test_loader = test_loader

        self.patience = 30
        self.scheduler = scheduler

        self.save_dir = save_dir or os.path.join("results", "model_saved", "best_model.pth")
        os.makedirs(os.path.dirname(self.save_dir), exist_ok=True)

        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        safe_tag = run_tag.replace("/", "-").replace(" ", "")
        prefix = f"{safe_tag}_" if safe_tag else ""

        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"training_log_{prefix}{self.start_time}.txt")


    def _forward_no_mix(self, weather, past, time, year, month, day, hour):
        batch_size = weather.size(0)
        ind = torch.arange(batch_size, device=device)
        lam = torch.ones(batch_size, 1, 1, 1, device=device)
        return self.model(weather, past, time, year, month, day, hour, ind, lam, Modal=0)

    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0

        print("-------------- start training --------------")

        with open(self.log_file, "w") as f:
            f.write(f"Training started at {self.start_time}\n")
            f.write(f"Epochs: {self.epochs}, Patience: {self.patience}\n")
            f.write(f"Model: {self.model}\n\n")
            if self.extra_log_header:
                f.write(self.extra_log_header.rstrip() + "\n")
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            train_samples = 0

            _sync_cuda()
            t0 = T.perf_counter()

            for weather, past, time, targets, year, month, day, hour in self.train_loader:
                weather, past, time, targets = (
                    weather.to(device),
                    past.to(device),
                    time.to(device),
                    targets.to(device),
                )
                year, month, day, hour = (
                    year.to(device),
                    month.to(device),
                    day.to(device),
                    hour.to(device),
                )

                train_samples += weather.size(0)

                outputs = self._forward_no_mix(weather, past, time, year, month, day, hour)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            _sync_cuda()
            train_time = T.perf_counter() - t0
            train_time_per_sample = train_time / max(train_samples, 1)
            train_samples_per_sec = train_samples / max(train_time, 1e-12)

            _sync_cuda()
            val_t0 = T.perf_counter()
            val_loss, val_samples = self.val()
            _sync_cuda()
            val_time = T.perf_counter() - val_t0
            val_time_per_sample = val_time / max(val_samples, 1)
            val_samples_per_sec = val_samples / max(val_time, 1e-12)
            epoch_time = train_time + val_time

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] | "
                f"Loss: {running_loss / len(self.train_loader):.7f} | "
                f"Val: {val_loss:.4f} | "
                f"train_time:{train_time:.2f}s | val_time:{val_time:.2f}s |  epoch_time:{epoch_time:.2f}s | "
                f"train_t/sample:{train_time_per_sample*1000:.2f}ms | val_t/sample:{val_time_per_sample*1000:.2f}ms"

            )

            with open(self.log_file, "a") as f:
                f.write(
                    f"Epoch [{epoch + 1}/{self.epochs}] | "
                    f"Loss: {running_loss / len(self.train_loader):.7f} | "
                    f"Val: {val_loss:.4f} | "
                    f"train_time:{train_time:.2f}s | val_time:{val_time:.2f}s |  epoch_time:{epoch_time:.2f}s | "
                    f"train_t/sample:{train_time_per_sample*1000:.2f}ms | val_t/sample:{val_time_per_sample*1000:.2f}ms\n"
                )
            if self.scheduler:
                self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_dir)
                print("Validation improved. Model saved.")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping.")
                    break

    def val(self):
        if self.val_loader is None:
            raise ValueError("val_loader is None.")

        self.model.eval()
        predictions, actuals = [], []
        val_samples = 0

        with torch.no_grad():
            for weather, past, time, targets, year, month, day, hour in self.val_loader:
                weather, past, time, targets = (
                    weather.to(device),
                    past.to(device),
                    time.to(device),
                    targets.to(device),
                )
                year, month, day, hour = (
                    year.to(device),
                    month.to(device),
                    day.to(device),
                    hour.to(device),
                )

                val_samples += weather.size(0)

                outputs = self._forward_no_mix(weather, past, time, year, month, day, hour)
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.cpu().numpy())

        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        val_loss = np.mean(np.abs(predictions - actuals))

        return val_loss, val_samples


class TrainerMixup:

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scaler_y,
        epochs,
        train_loader,
        val_loader,
        test_loader,
        scheduler=None,
        alpha=0.0,
        sigma=0.0,
        save_dir=None,
        run_tag: str = "",
        log_dir: str = "logs",
        extra_log_header: str = "",
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler_y = scaler_y
        self.epochs = epochs
        self.extra_log_header = extra_log_header

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.scheduler = scheduler
        self.alpha = alpha
        self.sigma = sigma
        self.patience = 30
        self.save_dir = save_dir

        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        safe_tag = run_tag.replace("/", "-").replace(" ", "")
        prefix = f"{safe_tag}_" if safe_tag else ""

        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"training_log_{prefix}{self.start_time}.txt")


    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0
        with open(self.log_file, "w") as f:
            f.write(f"Training started at {self.start_time}\n")
            f.write(f"Epochs: {self.epochs}, Patience: {self.patience}\n")
            f.write(f"Model: {self.model}\n\n")
            if self.extra_log_header:
                f.write(self.extra_log_header.rstrip() + "\n")
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            train_samples = 0

            _sync_cuda()
            t0 = T.perf_counter()

            for weather, past, time, targets, year, month, day, hour in self.train_loader:
                weather, past, time, targets = (
                    weather.to(device),
                    past.to(device),
                    time.to(device),
                    targets.to(device),
                )
                year, month, day, hour = (
                    year.to(device),
                    month.to(device),
                    day.to(device),
                    hour.to(device),
                )

                train_samples += weather.size(0)

                weather, past, time, targets, ind, lam = c_mixup(
                    weather, past, time, targets, alpha=self.alpha, sigma=self.sigma
                )

                outputs = self.model(weather, past, time, year, month, day, hour, ind, lam, Modal=1)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            _sync_cuda()
            train_time = T.perf_counter() - t0
            train_time_per_sample = train_time / max(train_samples, 1)
            train_samples_per_sec = train_samples / max(train_time, 1e-12)

            _sync_cuda()
            val_t0 = T.perf_counter()
            val_loss, val_samples = self.val()
            _sync_cuda()
            val_time = T.perf_counter() - val_t0
            val_time_per_sample = val_time / max(val_samples, 1)
            val_samples_per_sec = val_samples / max(val_time, 1e-12)
            epoch_time = train_time + val_time

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] | "
                f"Loss: {running_loss / len(self.train_loader):.7f} | "
                f"Val: {val_loss:.4f} | "
                f"train_time:{train_time:.2f}s | val_time:{val_time:.2f}s |  epoch_time:{epoch_time:.2f}s | "
                f"train_t/sample:{train_time_per_sample*1000:.2f}ms | val_t/sample:{val_time_per_sample*1000:.2f}ms"

            )

            with open(self.log_file, "a") as f:
                f.write(
                    f"Epoch [{epoch + 1}/{self.epochs}] | "
                    f"Loss: {running_loss / len(self.train_loader):.7f} | "
                    f"Val: {val_loss:.4f} | "
                    f"train_time:{train_time:.2f}s | val_time:{val_time:.2f}s |  epoch_time:{epoch_time:.2f}s | "
                    f"train_t/sample:{train_time_per_sample*1000:.2f}ms | val_t/sample:{val_time_per_sample*1000:.2f}ms\n"
                )

            if self.scheduler:
                self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_dir)
                print("Validation improved. Model saved.")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping.")
                    break

    def val(self):
        self.model.eval()
        predictions, actuals = [], []
        val_samples = 0

        with torch.no_grad():
            for weather, past, time, targets, year, month, day, hour in self.val_loader:
                weather, past, time, targets = (
                    weather.to(device),
                    past.to(device),
                    time.to(device),
                    targets.to(device),
                )
                year, month, day, hour = (
                    year.to(device),
                    month.to(device),
                    day.to(device),
                    hour.to(device),
                )

                val_samples += weather.size(0)

                batch_size = weather.size(0)
                ind = torch.arange(batch_size, device=device)
                lam = torch.ones(batch_size, 1, 1, 1, device=device)

                outputs = self.model(weather, past, time, year, month, day, hour, ind, lam, Modal=0)
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.cpu().numpy())

        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        val_loss = np.mean(np.abs(predictions - actuals))

        return val_loss, val_samples
