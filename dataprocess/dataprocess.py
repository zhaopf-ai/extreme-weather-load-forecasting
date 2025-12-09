import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pytest
from torchvision import transforms
import os
from datetime import datetime, timedelta
from PIL import Image, UnidentifiedImageError

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data_from_csv(file_path, test_size=0.1, batch_size=32):
    data = pd.read_csv(file_path)
    year = data['year'].values.reshape(-1, 1)
    month = data['month'].values.reshape(-1, 1)
    day = data['day'].values.reshape(-1, 1)
    hour = data['hour'].values.reshape(-1, 1)

    X = data.iloc[:, list(range(1, 11)) + list(range(15, 50))].values
    y = data.iloc[:, 0].values

    for i in range(X.shape[1]):
        X[:, i] = X[:, i]/np.max(X[:, i])

    X_weather = X[:, :4]
    X_past = X[:, 4:21]
    X_time = X[:, 21:]

    scaler_y = np.max(y)

    y = y/np.max(y)
    y = y.reshape(-1,1)

    X_w_train, X_w_test, X_p_train, X_p_test, X_t_train, X_t_test, y_train, y_test, year_train, year_test, month_train, month_test, day_train, day_test, hour_train, hour_test = train_test_split(
        X_weather, X_past, X_time, y, year, month, day, hour, test_size=test_size, shuffle=False)

    X_w_train = torch.tensor(X_w_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_w_test = torch.tensor(X_w_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    X_p_train = torch.tensor(X_p_train, dtype=torch.float32)
    X_p_test = torch.tensor(X_p_test, dtype=torch.float32)

    X_t_train = torch.tensor(X_t_train, dtype=torch.float32)
    X_t_test = torch.tensor(X_t_test, dtype=torch.float32)


    year_train = torch.tensor(year_train, dtype=torch.float32)
    year_test = torch.tensor(year_test, dtype=torch.float32)
    month_train = torch.tensor(month_train, dtype=torch.float32)
    month_test = torch.tensor(month_test, dtype=torch.float32)
    day_train = torch.tensor(day_train, dtype=torch.float32)
    day_test = torch.tensor(day_test, dtype=torch.float32)
    hour_train = torch.tensor(hour_train, dtype=torch.float32)
    hour_test = torch.tensor(hour_test, dtype=torch.float32)


    train_dataset = TensorDataset(X_w_train, X_p_train, X_t_train, y_train, year_train, month_train, day_train, hour_train)
    test_dataset = TensorDataset(X_w_test, X_p_test, X_t_test, y_test, year_test, month_test, day_test, hour_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler_y


class WeatherImageLoader:
    """Load and preprocess weather images with caching."""

    def __init__(self, img_dir, processed_dir, image_size=(224, 224)):
        self.img_dir = img_dir
        self.processed_dir = processed_dir
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4554, 0.4645, 0.3512],
                std=[0.1292, 0.1194, 0.1167]
            ),
        ])

        os.makedirs(self.processed_dir, exist_ok=True)

    def load_images_for_hour(self, timestamp):
        cached_name = f"cached_{timestamp.strftime('%Y%m%dT%H%M%S')}.pt"
        cached_path = os.path.join(self.processed_dir, cached_name)

        if os.path.exists(cached_path):
            return torch.load(cached_path)

        images = []
        for i in range(4):
            ts_offset = timestamp + timedelta(minutes=i * 15)
            img_name = f"Satellite-With-Radar_{ts_offset.strftime('%Y%m%dT%H%M%S')}+0800_modified.png"
            img_path = os.path.join(self.img_dir, img_name)

            if not os.path.exists(img_path):
                continue

            try:
                img = Image.open(img_path).convert('RGB')
                images.append(self.transform(img))
            except (OSError, IOError, UnidentifiedImageError):
                print(f"Skipped corrupted file: {img_path}")


            if len(images) != 0:
                aggregated_image = self.aggregate_images(torch.stack(images))
                torch.save(aggregated_image, cached_path)
                return aggregated_image
            else:
                transform = transforms.ToTensor()  
                img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), 'RGB')
                img = transform(img)
                torch.save(img, cached_path)
                return img

    def aggregate_images(self, image_tensor):

        return torch.mean(image_tensor, dim=0) if image_tensor is not None else None
