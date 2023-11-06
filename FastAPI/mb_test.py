import datetime
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from pymodbus.server.sync import StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
from pymodbus.datastore import ModbusSparseDataBlock
from joblib import load
from models import Informer
import requests
import struct

# x_stand = load('../scaler/scaler_x_stand_2input.joblib')
# y_stand = load('../scaler/scaler_y_stand_2input.joblib')
x_stand = load('../scaler/scaler_x_stand.joblib')
y_stand = load('../scaler/scaler_y_stand.joblib')
output_folder = "../result"  # 结果保存文件夹路径
model_path2 = "../weights/2informer_epoch_4.pth"
model_path = "../weights/informer_epoch_6.pth"
enc_in = 2
enc_in_eight = 8
s_len = 120
pre_len = 60
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化一个最大长度为120的双端队列
data_list = []

first = True


def two_registers_to_float(registers):
    """将两个16位整数转换为浮点数"""
    packed_value = struct.pack('>HH', *registers)
    float_value = struct.unpack('>f', packed_value)[0]
    return float_value


def load_model(model_path="weights/2informer_epoch_0.pth", pre_len=60, enc_in=2):
    model = Informer(out_len=pre_len, enc_in=enc_in)
    model.load_state_dict(
        torch.load(model_path))  # Load the trained model weights
    model.eval()
    model.to(device)
    return model


# model = load_model(model_path=model_path2, pre_len=pre_len, enc_in=enc_in)
model_8 = load_model(model_path=model_path, pre_len=pre_len, enc_in=enc_in_eight)


# 将你的推理函数定义在这里
def inference_streaming(model, test_x, test_y, batch_size=1):
    global first
    test_data = AmaData(test_x, test_y)
    test_data = DataLoader(test_data, shuffle=False, batch_size=1)
    predictions = []
    with torch.no_grad():
        for x, y, xt, yt in test_data:
            mask = torch.zeros_like(y)[:, pre_len:].to(device)
            x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
            dec_y = torch.cat([y[:, :pre_len], mask], dim=1)
            logits = model(x, xt, dec_y, yt)
            predictions.append(logits.cpu().numpy())  # Append predictions
            # if first:
            #     print(f'x shape: {x.shape}, x: {x}\n'
            #           f'dec_y shape: {dec_y.shape}, dec_y: {dec_y}\n'
            #           f'xt shape: {xt.shape}, xt: {xt}\n'
            #           f'yt shape: {yt.shape}, yt: {yt}\n'
            #           f'logits shape: {logits.shape}, logits: {logits}\n'
            #           )
            #     first = False

    original_labels = test_y[0, :, 1].astype(np.float32)
    predictions = np.concatenate(predictions, axis=0)
    original_predictions = y_stand.inverse_transform(predictions[:, :, 0]).flatten().astype(np.float32)
    predictions.tolist()
    original_predictions.tolist()
    return original_labels, original_predictions


def create_data(datas, pre_len = 60, s_len = 120):
    values = []
    labels = []

    datas = datas.values
    # for index in range(0, lens - pre_len - s_len):
    #     value = datas[index:index + s_len, [0, 1, 2]]
    #     label = datas[index + s_len - pre_len:index + s_len + pre_len, [0, 3]]
    #
    #     values.append(value)
    #     labels.append(label)

    value = datas[0:0 + s_len, [0, 1, 2]]
    label = datas[:s_len, [0, 3]]

    values.append(value)
    labels.append(label)

    # print('\n***************\nvalue', values)
    return values, labels


def read_data(datalist):
    global first
    feature_names = ['Time', 11, 13, 15]  # 与read_test_data中的特征名称相同
    dataframes = []

    for data in datalist:
        data_values = [data[feature_name] for feature_name in feature_names]
        datas = pd.DataFrame([data_values], columns=feature_names)  # 创建一个DataFrame来模拟CSV文件的结构
        # datas.pop(0)
        datas.fillna(0, inplace=True)  # 填充NaN值
        dataframes.append(datas)
    # 使用pd.concat()将所有DataFrame拼接起来
    concatenated_data = pd.concat(dataframes, ignore_index=True)
    # 获取concatenated_data[40015]列前60个值
    temp_labels = concatenated_data[15].values[pre_len:]
    concatenated_data[15].values[:pre_len] = temp_labels
    # print('\n***************\ndata:', concatenated_data)
    # concatenated_data.to_csv('test.csv')
    values, labels = create_data(concatenated_data, pre_len=pre_len, s_len=s_len)
    values = np.array(values)
    labels = np.array(labels)
    return values, labels


def create_data8(datas, pre_len = 60, s_len = 120):
    values = []
    labels = []

    datas = datas.values
    # for index in range(0, lens - pre_len - s_len):
    #     value = datas[index:index + s_len, [0, 1, 2]]
    #     label = datas[index + s_len - pre_len:index + s_len + pre_len, [0, 3]]
    #
    #     values.append(value)
    #     labels.append(label)

    value = datas[0:0 + s_len, [0, 1, 2, 3, 4, 5, 6, 7, 9]]
    label = datas[:s_len, [0, 8]]

    values.append(value)
    labels.append(label)

    # print('\n***************\nvalue', values)
    return values, labels


def read_data8(datalist):
    global first
    feature_names = ['Time', 1, 3, 5, 7, 9, 11, 13, 15, 17]  # 与read_test_data中的特征名称相同
    dataframes = []

    for data in datalist:
        data_values = [data[feature_name] for feature_name in feature_names]
        datas = pd.DataFrame([data_values], columns=feature_names)  # 创建一个DataFrame来模拟CSV文件的结构
        # datas.pop(0)
        datas.fillna(0, inplace=True)  # 填充NaN值
        dataframes.append(datas)
    # 使用pd.concat()将所有DataFrame拼接起来
    concatenated_data = pd.concat(dataframes, ignore_index=True)
    # 获取concatenated_data[40015]列前60个值
    temp_labels = concatenated_data[15].values[pre_len:]
    concatenated_data[15].values[:pre_len] = temp_labels
    # print('\n***************\ndata:', concatenated_data)
    # concatenated_data.to_csv('test.csv')
    values, labels = create_data8(concatenated_data, pre_len=pre_len, s_len=s_len)
    values = np.array(values)
    labels = np.array(labels)
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
        # print('\n***************\ntimedata:', time_data)
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


class CustomDataBlock(ModbusSparseDataBlock):
    def __init__(self, values):
        super().__init__(values)
        self.i = 0
        self.temp_values = {}  # 用于跟踪已更新的地址和它们的值

    def setValues(self, address, values):
        # 获取当前时间的时分秒
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        # 更新临时值
        print('\n***************\n接受数据:', values)

        # 将地址和值更新到临时值中
        for addr, value in zip(range(address, address + len(values)), values):
            self.temp_values[addr] = value

        # 检查是否收到了所有18个地址的数据
        if set(self.temp_values.keys()) == set(initial_values.keys()):
            # 将整组数据作为一个字典存储在data_list中
            # 组合浮点数并添加到data_list
            for addr in range(1, 18, 2):  # 遍历9组地址
                float_value = two_registers_to_float([self.temp_values[addr], self.temp_values[addr + 1]])
                print('\n***************\n接受数据转化:', float_value)
                self.temp_values[addr] = float_value
            my_dict = {'Time': current_time}
            my_dict.update(self.temp_values)  # 将self.temp_values合并到my_dict中
            # print(f'my_dict:{my_dict}')
            data_list.append(my_dict)
            # print('\n***************\ndatalist:', data_list)
            # print('\n***************\ndatalist长度:', len(data_list))
            self.temp_values.clear()  # 清空临时值
            if len(data_list) > 120:
                data_list.pop(0)  # 从队列左侧删除旧数据
            if len(data_list) == 120:

                # 还要改前面47行左右的model和开头的x、y_stand
                # test_x, test_y = read_data(data_list)
                test_x, test_y = read_data8(data_list)

                # 调用推理函数，逐步生成预测结果
                # original_labels, original_predictions = inference_streaming(model, test_x, test_y, batch_size=1)
                original_labels, original_predictions = inference_streaming(model_8, test_x, test_y, batch_size=1)

                print(f'original_labels:{original_labels[pre_len:]}')
                print(f'original_predictions:{original_predictions}')
                predictions_generator = [original_labels[pre_len:].tolist(), original_predictions.tolist()]
                print(f'predictions_generator:{predictions_generator}')
                # print('\n***************\nresult转化:', result_list)
                try:
                    response = requests.post("http://localhost:7860/data/", json={"data": predictions_generator})
                    # response = requests.post("http://47.109.83.116:7860/data/", json={"data": predictions_generator})
                    response.raise_for_status()  # 检查是否有错误状态
                    print('***************\nresponse:', response)
                except requests.exceptions.RequestException as e:
                    print('POST请求发生异常:', e)

        super().setValues(address, values)


# 根据您的需求，初始化数据块
initial_values = {
    1: 0,  # 脱硫岛入口烟气SO2折算浓度
    2: 0,  # 脱硫岛入口烟气SO2折算浓度
    3: 0,  # 脱硫岛入口烟气干标流量
    4: 0,  # 脱硫岛入口烟气干标流量
    5: 0,  # 吸收塔床层压降
    6: 0,  # 吸收塔床层压降
    7: 0,  # 吸收塔入口烟气温度
    8: 0,  # 吸收塔入口烟气温度
    9: 0,  # 吸收塔出口烟气温度均值
    10: 0,  # 吸收塔出口烟气温度均值
    11: 0,  # 消石灰调频旋转给料器1频率
    12: 0,  # 消石灰调频旋转给料器1频率
    13: 0,  # 消石灰调频旋转给料器2频率
    14: 0,  # 消石灰调频旋转给料器2频率
    15: 0,  # 布袋出口烟气SO2折算浓度
    16: 0,  # 布袋出口烟气SO2折算浓度
    17: 0,  # 出口SO2控制设定值
    18: 0,  # 出口SO2控制设定值
    # 40019: 0  # 出口SO2模型预测值
}

# 使用自定义的数据块初始化保持寄存器
store = ModbusSlaveContext(
    hr=CustomDataBlock(initial_values)
)

context = ModbusServerContext(slaves=store, single=True)
print('server start')
# 启动服务器
StartTcpServer(context, address=('127.0.0.1', 502))
