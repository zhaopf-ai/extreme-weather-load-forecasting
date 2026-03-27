import os
import time as T
from datetime import datetime
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync_cuda():
    if device.type == "cuda":
        torch.cuda.synchronize()


def _parse_model_output(outputs):
    if isinstance(outputs, tuple):
        if len(outputs) != 2:
            raise ValueError("Model output tuple must be (pred, extra_dict).")
        pred, extra = outputs
        if extra is None:
            extra = {}
        return pred, extra
    return outputs, {}


def _get_aux_loss(extra, device_):
    if not isinstance(extra, dict):
        return torch.tensor(0.0, device=device_)

    if "total_aux_loss" in extra and extra["total_aux_loss"] is not None:
        aux_loss = extra["total_aux_loss"]
        if not torch.is_tensor(aux_loss):
            aux_loss = torch.tensor(float(aux_loss), device=device_)
        return aux_loss

    if "dcca_loss" in extra and extra["dcca_loss"] is not None:
        aux_loss = extra["dcca_loss"]
        if not torch.is_tensor(aux_loss):
            aux_loss = torch.tensor(float(aux_loss), device=device_)
        return aux_loss

    return torch.tensor(0.0, device=device_)


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

        self.patience = 20
        self.scheduler = scheduler

        self.save_dir = save_dir or os.path.join("results", "model_saved", "best_model.pth")
        os.makedirs(os.path.dirname(self.save_dir), exist_ok=True)

        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        safe_tag = run_tag.replace("/", "-").replace(" ", "")
        prefix = f"{safe_tag}_" if safe_tag else ""

        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"training_log_{prefix}{self.start_time}.txt")

    def _forward_no_mix(self, weather, past, daily, weekly, time, year, month, day, hour):
        batch_size = weather.size(0)
        ind = torch.arange(batch_size, device=device)
        lam = torch.ones(batch_size, 1, 1, 1, device=device)
        return self.model(weather, past, daily, weekly, time, year, month, day, hour, ind, lam, Modal=0, return_extra=True)

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
            running_total_loss = 0.0
            running_pred_loss = 0.0
            running_aux_loss = 0.0
            train_samples = 0

            _sync_cuda()
            t0 = T.perf_counter()

            for weather, past, daily, weekly, time, targets, year, month, day, hour in self.train_loader:
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

                train_samples += weather.size(0)

                outputs = self._forward_no_mix(weather, past, daily, weekly, time, year, month, day, hour)
                preds, extra = _parse_model_output(outputs)

                pred_loss = self.criterion(preds, targets)
                aux_loss = _get_aux_loss(extra, preds.device)
                total_loss = pred_loss + aux_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                running_total_loss += total_loss.item()
                running_pred_loss += pred_loss.item()
                running_aux_loss += aux_loss.item()

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

            msg = (
                f"Epoch [{epoch + 1}/{self.epochs}] | "
                f"Total: {running_total_loss / len(self.train_loader):.7f} | "
                f"Pred: {running_pred_loss / len(self.train_loader):.7f} | "
                f"Aux: {running_aux_loss / len(self.train_loader):.7f} | "
                f"Val: {val_loss:.4f} | "
                f"train_time:{train_time:.2f}s | "
                f"val_time:{val_time:.2f}s | "
                f"epoch_time:{epoch_time:.2f}s | "
                f"train_t/sample:{train_time_per_sample * 1000:.2f}ms | "
                f"val_t/sample:{val_time_per_sample * 1000:.2f}ms"
            )
            print(msg)

            with open(self.log_file, "a") as f:
                f.write(msg + "\n")

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
            for weather, past, daily, weekly, time, targets, year, month, day, hour in self.val_loader:
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

                val_samples += weather.size(0)

                outputs = self._forward_no_mix(weather, past, daily, weekly, time, year, month, day, hour)
                preds, _ = _parse_model_output(outputs)

                predictions.append(preds.detach().cpu().numpy())
                actuals.append(targets.detach().cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        val_loss = np.mean(np.abs(predictions - actuals))

        return val_loss, val_samples
