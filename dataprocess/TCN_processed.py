import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
import os
from datetime import timedelta
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data_from_csv(file_path, windows_size=10, test_size=0.1, batch_size=32):
    data = pd.read_csv(file_path)
    year = data['year'].values.reshape(-1, 1)
    month = data['month'].values.reshape(-1, 1)
    day = data['day'].values.reshape(-1, 1)
    hour = data['hour'].values.reshape(-1, 1)

    y = data.iloc[:, 0].values  # 目标
    X = []
    for i in range(windows_size + 1, len(y)):
        tmp = []
        for j in range(1, windows_size + 1):
            tmp.append(y[i - j])
        X.append(tmp)
    X = np.array(X)
    y = [y[i] for i in range(windows_size + 1, len(y))]
    y = np.array(y)

    year = year[windows_size + 1:]
    month = month[windows_size + 1:]
    day = day[windows_size + 1:]
    hour = hour[windows_size + 1:]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1))

    X = X.reshape(len(X), 1, -1)

    X_train, X_test, y_train, y_test, year_train, year_test, month_train, month_test, day_train, day_test, hour_train, hour_test = train_test_split(
        X, y, year, month, day, hour, test_size=test_size, shuffle=False)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)


    year_train = torch.tensor(year_train, dtype=torch.float32)
    year_test = torch.tensor(year_test, dtype=torch.float32)
    month_train = torch.tensor(month_train, dtype=torch.float32)
    month_test = torch.tensor(month_test, dtype=torch.float32)
    day_train = torch.tensor(day_train, dtype=torch.float32)
    day_test = torch.tensor(day_test, dtype=torch.float32)
    hour_train = torch.tensor(hour_train, dtype=torch.float32)
    hour_test = torch.tensor(hour_test, dtype=torch.float32)
    train_dataset = TensorDataset(X_train, y_train, year_train, month_train, day_train, hour_train)
    test_dataset = TensorDataset(X_test, y_test, year_test, month_test, day_test, hour_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader, X_test, y_test, scaler_y


class WeatherImageLoader:
    def __init__(self, img_dir, processed_dir='data\\processed_image', image_size=(224, 224)):
        self.img_dir = img_dir
        self.processed_dir = processed_dir
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        ])

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def load_images_for_hour(self, timestamp):
        images = []
        for i in range(4):
            minute_offset = i * 15
            timestamp_offset = timestamp + timedelta(minutes=minute_offset)
            img_filename = f"Satellite-With-Radar_{timestamp_offset.strftime('%Y%m%dT%H%M%S')}+0800_modified.png"
            img_path = os.path.join(self.img_dir, img_filename)

            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = self.transform(img)
                    images.append(img)
                except (OSError, IOError):
                    transform = transforms.ToTensor()
                    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), 'RGB')
                    img = transform(img)
                    images.append(img)
            else:
                transform = transforms.ToTensor()
                img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), 'RGB')
                img = transform(img)
                images.append(img)

        return torch.concatenate(images, dim=0)








