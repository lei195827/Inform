import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from models import Informer
import matplotlib.pyplot as plt

train_size = 0.85
x_stand = StandardScaler()
y_stand = StandardScaler()
s_len = 120
pre_len = 30
batch_size = 1
device = "cuda"
lr = 5e-5
epochs = 10


def create_data(datas):
    values = []
    labels = []

    lens = datas.shape[0]
    datas = datas.values
    for index in range(0, lens - pre_len - s_len):
        value = datas[index:index + s_len, [0, 1, 2, 3, 4, 5, 6, 7, 9]]
        label = datas[index + s_len - pre_len:index + s_len + pre_len, [0, 8]]

        values.append(value)
        labels.append(label)

    return values, labels


def read_test_data(file_path):
    datas = pd.read_csv(file_path)
    datas.pop("Unnamed: 0")
    datas.pop("序号")
    datas.fillna(0)
    # print(datas)
    xs = datas.values[:, [1, 2, 3, 4, 5, 6, 7, 9]]  # value
    ys = datas.values[:, [8]]  # label

    x_stand.fit(xs)
    # y_stand.fit(ys[:, None])
    y_stand.fit(ys)

    values, labels = create_data(datas)
    values = np.array(values)  # Convert to NumPy array
    labels = np.array(labels)  # Convert to NumPy array

    return values, labels


# 自定义数据集
class AmaData(Dataset):
    def __init__(self, values, labels):
        self.values, self.labels = values, labels

    def __len__(self):
        return len(self.values)

    def create_time(self, data):
        # 将时间戳字符串按空格进行分割
        # print("源数据：", data[0])
        time = data[:, 0]
        # print(time)
        hours = np.array([])
        minutes = np.array([])
        seconds = np.array([])
        for date_time in time:
            # 将时间部分再按冒号进行分割
            date_time = str(date_time)
            date_time = date_time.split(":")
            # 分别提取出时、分、秒
            hour = int(date_time[0])  # 0
            minute = int(date_time[1])  # 0
            second = int(date_time[2])  # 0

            # 将提取出的时、分、秒存入对应的数组
            hours = np.append(hours, hour)
            minutes = np.append(minutes, minute)
            seconds = np.append(seconds, second)
        hours = np.int32(hours)[:, None]
        minutes = np.int32(minutes)[:, None]
        seconds = np.int32(seconds)[:, None]
        time_data = np.concatenate([hours, minutes, seconds], axis=-1)
        # print(time_data)
        return time_data

    def __getitem__(self, item):
        value = self.values[item]
        label = self.labels[item]

        value_t = self.create_time(value)
        label_t = self.create_time(label)

        value = x_stand.transform(value[:, 1:])
        label = y_stand.transform(label[:, [1]])
        value = np.float32(value)
        label = np.float32(label)
        return value, label, value_t, label_t


def predict_and_visualize(file_path="datas/demo/f(t) 趋势视图_样式.csv", model_path="weights/informer.pth"):
    test_x, test_y = read_test_data(file_path)  # Replace with function to read test data

    test_data = AmaData(test_x, test_y)
    test_data = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    model = Informer()
    model.load_state_dict(
        torch.load(model_path))  # Load the trained model weights
    model.eval()
    model.to(device)

    predictions = []

    with torch.no_grad():
        for x, y, xt, yt in test_data:
            mask = torch.zeros_like(y)[:, pre_len:].to(device)

            x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
            dec_y = torch.cat([y[:, :pre_len], mask], dim=1)

            logits = model(x, xt, dec_y, yt)
            predictions.append(logits.cpu().numpy())  # Append predictions

    predictions = np.concatenate(predictions, axis=0)

    # Inverse transform the predictions and true labels
    # original_predictions = y_stand.inverse_transform(predictions[:, :, 0])
    # original_labels = test_y[:, pre_len:, 1]
    original_predictions = y_stand.inverse_transform(predictions[:, :, 0])
    original_predictions = original_predictions[:, -1]
    print(f"original_predictions.shape:{original_predictions.shape}")
    original_labels = test_y[:, -1, 1]
    print(f'original_labels.shape:{original_labels.shape}')
    print("original_labels:", original_labels)
    print("original_predictions:", original_predictions)

    # Plot and save the predictions and true values
    plt.figure(figsize=(10, 6))
    # plt.plot(original_predictions[:500,], label='Predicted')
    # plt.plot(original_labels[:500,], label='True')
    plt.plot(original_predictions, label='Predicted')
    plt.plot(original_labels, label='True')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title(f'Prediction vs True - Sample')
    plt.legend()
    plt.savefig(f'prediction_true_sample.png')
    plt.show()


if __name__ == '__main__':
    predict_and_visualize(file_path="datas/demo/f(t) 趋势视图_22.6.24~28_new.csv")
