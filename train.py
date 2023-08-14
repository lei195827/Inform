import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from models import Informer

train_size = 0.85
x_stand = StandardScaler()
y_stand = StandardScaler()
s_len = 180
pre_len = 60
batch_size = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def read_data():
    datas = pd.read_csv("./datas/demo/f(t) 趋势视图_22.6.9~13_new.csv")
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

    train_x, test_x, train_y, test_y = train_test_split(values, labels, train_size=train_size)

    return train_x, test_x, train_y, test_y


def read_data_from_folder(folder_path="datas/demo", train_size=0.8):
    all_datas = []

    # 遍历文件夹中的所有CSV文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            print(f"filename:{filename}")
            file_path = os.path.join(folder_path, filename)
            datas = pd.read_csv(file_path)
            datas.pop("Unnamed: 0")
            datas.pop("序号")
            datas.fillna(0, inplace=True)
            all_datas.append(datas)

    # 将所有CSV数据合并为一个DataFrame
    merged_datas = pd.concat(all_datas, ignore_index=True)
    print(f"merged_datas:{merged_datas.shape}")

    xs = merged_datas.values[:, [1, 2, 3, 4, 5, 6, 7, 9]]
    ys = merged_datas.values[:, [8]]


    x_stand.fit(xs)
    # y_stand.fit(ys[:, None])
    y_stand.fit(ys)

    values, labels = create_data(merged_datas)

    train_x, test_x, train_y, test_y = train_test_split(values, labels, train_size=train_size)

    return train_x, test_x, train_y, test_y


# 自定义数据集
class AmaData(Dataset):
    def __init__(self, values, labels):
        self.values, self.labels = values, labels

    def __len__(self):
        return len(self.values)

    # def create_time(self, data):
    #     time = data[:, 0]
    #     time = pd.to_datetime(time)
    #
    #     week = np.int32(time.dayofweek)[:, None]
    #     month = np.int32(time.month)[:, None]
    #     day = np.int32(time.day)[:, None]
    #     time_data = np.concatenate([month, week, day], axis=-1)
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


def train():
    train_x, test_x, train_y, test_y = read_data_from_folder(folder_path="datas/demo3")

    train_data = AmaData(train_x, train_y)
    train_data = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    test_data = AmaData(test_x, test_y)
    test_data = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    model = Informer(out_len=pre_len)
    model.train()
    model.to(device)

    loss_fc = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        pbar = tqdm(train_data)
        for step, (x, y, xt, yt) in enumerate(pbar):
            mask = torch.zeros_like(y)[:, pre_len:].to(device)

            x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
            dec_y = torch.cat([y[:, :pre_len], mask], dim=1)

            logits = model(x, xt, dec_y, yt)
            print(f"logits.shape:{logits.shape}")
            print(f"y.shape:{y.shape}")
            print(f"y[:, pre_len:]:{y[:, pre_len:].shape}")
            loss = loss_fc(logits, y[:, pre_len:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            s = "train ==> epoch:{} - step:{} - loss:{}".format(epoch, step, loss)

            pbar.set_description(s)

        model.eval()
        with torch.no_grad():
            pbar = tqdm(test_data)
            for step, (x, y, xt, yt) in enumerate(pbar):
                mask = torch.zeros_like(y)[:, pre_len:].to(device)

                x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
                dec_y = torch.cat([y[:, :pre_len], mask], dim=1)

                logits = model(x, xt, dec_y, yt)

                loss = loss_fc(logits, y[:, pre_len:])

                s = "test ==> epoch:{} - step:{} - loss:{}".format(epoch, step, loss)

                pbar.set_description(s)

        model.train()
        # 每2个epoch保存一次权重
        if epoch % 2 == 0:
            torch.save(model.state_dict(), f"weights/informer_epoch_{epoch}.pth")
            print(f'save model success')


if __name__ == '__main__':
    train()
