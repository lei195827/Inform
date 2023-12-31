import os

import torch
from joblib import load
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from thop import profile
from models import Informer
import matplotlib.pyplot as plt

x_stand = load('scaler/scaler_x_stand.joblib')
y_stand = load('scaler/scaler_y_stand.joblib')
input_folder = "datas/val"  # 文件夹路径
output_folder = "result2"  # 结果保存文件夹路径
model_path = "weights/informer_epoch_8.pth"
s_len = 120
enc_in = 8
pre_len = 60
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_data(datas, s_len=120, pre_len=60):
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
    # feature_names = ['序号', "出口SO2控制设定值", "脱硫岛入口烟气SO2折算浓度", "脱硫岛入口烟气干标流量",
    #                  "Unnamed: 0", "吸收塔床层压降", "吸收塔入口烟气温度", "吸收塔出口烟气温度"]
    feature_names = ['序号', "Unnamed: 0"]
    for feature_name in feature_names:
        datas.pop(str(feature_name))
    datas.fillna(0)
    # 测试当两个旋转角频率置0的影响
    # datas.iloc[:, 1:3] = 0
    values, labels = create_data(datas, s_len=s_len, pre_len=pre_len)
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


def insert_predictions_to_csv(file_path, n, original_predictions):
    # 读取原始 CSV 文件
    df = pd.read_csv(file_path)

    # 提取原文件名
    file_name = os.path.basename(file_path)
    # 删除扩展名
    file_name = file_name.split(".")[0]

    # 构造新的列名
    new_column_name = f"{file_name}_predictions"

    # 在 DataFrame 中插入新的列并设置预测值
    df[new_column_name] = None
    df.iloc[n:, -1] = original_predictions

    # 构造新的文件名
    new_file_name = f"{file_name}_with_predictions.csv"
    new_file_name = os.path.join("result2/csv", new_file_name)

    # 保存修改后的数据到新的 CSV 文件
    df.to_csv(new_file_name, index=False)
    print(f"New CSV file saved as {new_file_name}.")


def predict_and_visualize(file_path="datas/demo/f(t) 趋势视图_样式.csv", model_path="weights/2informer_epoch_0.pth",
                          save_file_name="prediction_true_sample.png"):
    test_x, test_y = read_test_data(file_path)  # Replace with function to read test data

    test_data = AmaData(test_x, test_y)
    test_data = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    model = Informer(out_len=pre_len, enc_in=enc_in)
    model.load_state_dict(
        torch.load(model_path))  # Load the trained model weights
    model.eval()
    model.to(device)

    # For demonstration, taking a single batch of data for FLOPs calculation
    x_sample, y_sample, xt_sample, yt_sample = next(iter(test_data))
    x_sample, xt_sample, yt_sample = x_sample.to(device), xt_sample.to(device), yt_sample.to(device)
    dec_y_sample = torch.cat([y_sample[:, :pre_len], torch.zeros_like(y_sample)[:, pre_len:]], dim=1).to(device)

    # Calculating and printing FLOPs before entering the prediction loop
    macs, params = profile(model, inputs=(x_sample, xt_sample, dec_y_sample, yt_sample), verbose=False)
    print(f"Total FLOPs: {macs * 2:.2f}")

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
    original_labels = test_y[:, -1, 1]
    # Calculate and print the average loss
    loss_fc = nn.L1Loss(reduction='mean')  # 使用均方误差作为损失函数
    original_predictions = original_predictions.astype(np.float32)
    original_labels = original_labels.astype(np.float32)
    average_loss = loss_fc(torch.tensor(original_predictions, dtype=torch.float32),
                           torch.tensor(original_labels, dtype=torch.float32)).item()
    print(f'Average Loss: {average_loss:.4f}')

    # Plot and save the predictions and true values
    plt.figure(figsize=(10, 6))
    print(f"test_x.shape:{test_x.shape}"
          f"test_y.shape:{test_y.shape}"
          f"predictions.shape:{original_labels.shape}"
          f"original_predictions.shape:{original_predictions.shape}")
    insert_predictions_to_csv(file_path=file_path, n=s_len + pre_len, original_predictions=original_predictions)
    plt.plot(original_predictions, label='Predicted')
    plt.plot(original_labels, label='True')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title(f'Prediction vs True - Sample')
    plt.legend()
    plt.figtext(0.95, 0.95, f'Average Loss: {average_loss:.4f}', ha='right', va='top')
    plt.savefig(save_file_name)
    plt.close()


def process_directory(input_folder, output_folder, model_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"prediction_true_{filename}.png")
            predict_and_visualize(file_path, model_path, output_path)
            print(f"Processed {filename} and saved result to {output_path}")


if __name__ == '__main__':
    # predict_and_visualize(file_path="datas/val/table_1.csv", model_path="weights/informer_epoch_4.pth")
    process_directory(input_folder, output_folder, model_path)
