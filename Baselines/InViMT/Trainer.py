import os
import time as T
from datetime import datetime

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync_cuda():
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
        patience: int = 10,
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

        self.scheduler = scheduler
        self.patience = patience

        self.save_dir = save_dir or os.path.join("results", "model_saved", "best_model.pth")
        os.makedirs(os.path.dirname(self.save_dir), exist_ok=True)

        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        safe_tag = run_tag.replace("/", "-").replace(" ", "")
        prefix = f"{safe_tag}_" if safe_tag else ""

        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"training_log_{prefix}{self.start_time}.txt")

    def _move_batch_to_device(self, batch):
        weather, past, daily, weekly, time_feat, targets, year, month, day, hour = batch

        weather = weather.to(device)
        past = past.to(device)
        daily = daily.to(device)
        weekly = weekly.to(device)
        time_feat = time_feat.to(device)
        targets = targets.to(device)

        year = year.to(device)
        month = month.to(device)
        day = day.to(device)
        hour = hour.to(device)

        return weather, past, daily, weekly, time_feat, targets, year, month, day, hour

    def _forward(self, weather, past, daily, weekly, time_feat, year, month, day, hour):
        return self.model(
            weather,
            past,
            daily,
            weekly,
            time_feat,
            year,
            month,
            day,
            hour,
        )

    def train(self):
        if self.val_loader is None:
            raise ValueError("val_loader is None.")

        best_val_loss = float("inf")
        patience_counter = 0

        print("-------------- start training --------------")

        with open(self.log_file, "w", encoding="utf-8") as f:
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

            for batch in self.train_loader:
                weather, past, daily, weekly, time_feat, targets, year, month, day, hour = self._move_batch_to_device(batch)

                train_samples += weather.size(0)

                outputs = self._forward(
                    weather,
                    past,
                    daily,
                    weekly,
                    time_feat,
                    year,
                    month,
                    day,
                    hour,
                )

                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            _sync_cuda()
            train_time = T.perf_counter() - t0
            train_time_per_sample = train_time / max(train_samples, 1)

            _sync_cuda()
            val_t0 = T.perf_counter()
            val_loss, val_samples = self.val()
            _sync_cuda()
            val_time = T.perf_counter() - val_t0
            val_time_per_sample = val_time / max(val_samples, 1)

            epoch_time = train_time + val_time
            train_loss_epoch = running_loss / max(len(self.train_loader), 1)

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] | "
                f"Loss: {train_loss_epoch:.7f} | "
                f"Val: {val_loss:.7f} | "
                f"train_time:{train_time:.2f}s | "
                f"val_time:{val_time:.2f}s | "
                f"epoch_time:{epoch_time:.2f}s | "
                f"train_t/sample:{train_time_per_sample * 1000:.2f}ms | "
                f"val_t/sample:{val_time_per_sample * 1000:.2f}ms"
            )

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"Epoch [{epoch + 1}/{self.epochs}] | "
                    f"Loss: {train_loss_epoch:.7f} | "
                    f"Val: {val_loss:.7f} | "
                    f"train_time:{train_time:.2f}s | "
                    f"val_time:{val_time:.2f}s | "
                    f"epoch_time:{epoch_time:.2f}s | "
                    f"train_t/sample:{train_time_per_sample * 1000:.2f}ms | "
                    f"val_t/sample:{val_time_per_sample * 1000:.2f}ms\n"
                )

            if self.scheduler is not None:
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
        predictions = []
        actuals = []
        val_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                weather, past, daily, weekly, time_feat, targets, year, month, day, hour = self._move_batch_to_device(batch)

                val_samples += weather.size(0)

                outputs = self._forward(
                    weather,
                    past,
                    daily,
                    weekly,
                    time_feat,
                    year,
                    month,
                    day,
                    hour,
                )

                predictions.append(outputs.detach().cpu().numpy())
                actuals.append(targets.detach().cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)

        val_loss = np.mean(np.abs(predictions - actuals))
        return val_loss, val_samples

    def test(self, test_loader=None):
        loader = test_loader if test_loader is not None else self.test_loader
        if loader is None:
            raise ValueError("test_loader is None.")

        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch in loader:
                weather, past, daily, weekly, time_feat, targets, year, month, day, hour = self._move_batch_to_device(batch)

                outputs = self._forward(
                    weather,
                    past,
                    daily,
                    weekly,
                    time_feat,
                    year,
                    month,
                    day,
                    hour,
                )

                predictions.append(outputs.detach().cpu().numpy())
                actuals.append(targets.detach().cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)

        test_loss = np.mean(np.abs(predictions - actuals))
        return test_loss, predictions, actuals