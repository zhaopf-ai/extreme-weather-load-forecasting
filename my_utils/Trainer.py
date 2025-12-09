import torch
import numpy as np
from datetime import datetime
from my_utils.helper import compute_mape
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from my_utils.mixup import c_mixup
from fvcore.nn import FlopCountAnalysis
import time as T



class Trainer:
    def __init__(self, model, criterion, optimizer, scaler_y, epochs, train_loader, test_loader, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler_y = scaler_y
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.patience = 40
        self.scheduler = scheduler

        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = f'logs/training_log_{self.start_time}.txt'
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0
        self.model.train()
        print("--------------start training!--------------")

        with open(self.log_file, 'w') as f:  # 打开日志文件写入
            f.write(f"Training started at {self.start_time}\n")
            f.write(f"Epochs: {self.epochs}, Patience: {self.patience}\n")
            f.write(f"Model architecture: {str(self.model)}\n")
            f.write("\n")


        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (weather, past, time, targets, year, month, day, hour) in enumerate(self.train_loader):
                weather, past, time, targets= weather.to(device), past.to(device), time.to(device), targets.to(device)
                year, month, day, hour = year.to(device), month.to(device), day.to(device), hour.to(device)
                outputs = self.model(weather, past, time, year, month, day, hour)

                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            val_loss = self.val()
            print(
                f'Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss / len(self.train_loader):.7f},mape:{val_loss}')
            with open(self.log_file, 'a') as f:
                f.write(
                    f'Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss / len(self.train_loader):.7f}, mape: {val_loss}\n')
            if self.scheduler:
                self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print("Validation loss decreased. Saving model...")
                torch.save(self.model.state_dict(), 'results\\model_saved\\best_model.pth')

            else:
                patience_counter += 1
                print(f'No improvement in validation loss for {patience_counter} epochs.')
                if patience_counter >= self.patience:
                    print("Early stopping triggered!")
                    break

    def test(self, model):
        model.eval()
        print("--------------start testing!--------------")
        predictions, actuals = [], []
        with torch.no_grad():
            for weather, past, time, targets, year, month, day, hour in self.test_loader:
                weather, past, time, targets= weather.to(device), past.to(device), time.to(device), targets.to(device)
                year, month, day, hour = year.to(device), month.to(device), day.to(device), hour.to(device)
                outputs = model(weather, past, time, year, month, day, hour)
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.cpu().numpy())

        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)

        return predictions, actuals

    def val(self):
        self.model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            for weather, past, time, targets, year, month, day, hour in self.test_loader:
                weather, past, time, targets= weather.to(device), past.to(device), time.to(device), targets.to(device)
                year, month, day, hour = year.to(device), month.to(device), day.to(device), hour.to(device)
                outputs = self.model(weather, past, time, year, month, day, hour)
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.cpu().numpy())

        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)

        mape = compute_mape(predictions, actuals, self.scaler_y)

        return mape


class TrainerMixup:
    def __init__(self, model, criterion, optimizer, scaler_y, epochs, train_loader, test_loader, scheduler=None, alpha=0, sigma=0, save_dir=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler_y = scaler_y
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.patience = 40
        self.scheduler = scheduler

        self.alpha = alpha
        self.sigma = sigma
        self.save_dir = save_dir

        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = f'logs/training_log_{self.start_time}.txt'
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0
        self.model.train()
        print("--------------start training!--------------")

        with open(self.log_file, 'w') as f:  # 打开日志文件写入
            f.write(f"Training started at {self.start_time}\n")
            f.write(f"Epochs: {self.epochs}, Patience: {self.patience}\n")
            f.write(f"Model architecture: {str(self.model)}\n")
            f.write("\n")


        for epoch in range(self.epochs):
            running_loss = 0.0
            epoch_start_time = T.time()

            for i, (weather, past, time, targets, year, month, day, hour) in enumerate(self.train_loader):
                weather, past, time, targets= weather.to(device), past.to(device), time.to(device), targets.to(device)
                year, month, day, hour = year.to(device), month.to(device), day.to(device), hour.to(device)

                weather, past, time, targets, ind, lam = c_mixup(weather, past, time, targets, alpha=self.alpha, sigma=self.sigma)


                outputs = self.model(weather, past, time, year, month, day, hour, ind, lam, Modal=1)

                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            val_loss = self.val()
            epoch_end_time = T.time()
            epoch_duration = epoch_end_time - epoch_start_time

            print(
                f'Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss / len(self.train_loader):.7f},mape:{val_loss},Time: {epoch_duration:.2f} seconds')


            with open(self.log_file, 'a') as f:
                f.write(
                    f'Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss / len(self.train_loader):.7f}, mape: {val_loss}\n')


            if self.scheduler:
                self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print("Validation loss decreased. Saving model...")
                torch.save(self.model.state_dict(), self.save_dir)

            else:
                patience_counter += 1
                print(f'No improvement in validation loss for {patience_counter} epochs.')
                if patience_counter >= self.patience:
                    print("Early stopping triggered!")
                    break

    def test(self, model):
        model.eval()
        print("--------------start testing!--------------")
        predictions, actuals = [], []
        with torch.no_grad():
            for weather, past, time, targets, year, month, day, hour in self.test_loader:
                weather, past, time, targets = weather.to(device), past.to(device), time.to(device), targets.to(device)
                year, month, day, hour = year.to(device), month.to(device), day.to(device), hour.to(device)

                batch_size = weather.size(0)
                ind = torch.arange(batch_size, device=device)
                lam = torch.ones(batch_size, 1, 1, 1, device=device)

                outputs = model(weather, past, time, year, month, day, hour, ind, lam, Modal=0)
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.cpu().numpy())

        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)

        return predictions, actuals

    def val(self):
        self.model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            for weather, past, time, targets, year, month, day, hour in self.test_loader:
                weather, past, time, targets = weather.to(device), past.to(device), time.to(device), targets.to(device)
                year, month, day, hour = year.to(device), month.to(device), day.to(device), hour.to(device)

                batch_size = weather.size(0)
                ind = torch.arange(batch_size, device=device)
                lam = torch.ones(batch_size, 1, 1, 1, device=device)

                outputs = self.model(weather, past, time, year, month, day, hour, ind, lam, Modal=0)
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.cpu().numpy())

        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)

        mape = compute_mape(predictions, actuals, self.scaler_y)

        return mape
