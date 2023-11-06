import asyncio
import datetime
from typing import List
from fastapi import Request, status
from fastapi.responses import JSONResponse
import pandas as pd
import uvicorn
from fastapi import WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import logging
import datetime
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from joblib import load
from models import Informer
import httpx

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MyLogger')

# 初始化FastAPI应用
app = FastAPI()

# 挂载静态文件夹
app.mount("/node_modules", StaticFiles(directory="node_modules"), name="node_modules")

# 设置模板目录
templates = Jinja2Templates(directory="templates")

# 初始化数据队列和其他相关数据
history_predictions = [None] * 120
predictions = [None] * 60
predictions.extend(np.zeros(60).tolist())
true_values = np.zeros(60).tolist()
true_values.extend([None] * 60)
send_data = np.zeros((4, 120)).tolist()
# 用于推理的字典列表
data_list = []
# 全局的WebSocket连接集合
connected_websockets = set()


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

    original_labels = test_y[0, :, 1].astype(np.float32)
    predictions = np.concatenate(predictions, axis=0)
    original_predictions = y_stand.inverse_transform(predictions[:, :, 0]).flatten().astype(np.float32)
    predictions.tolist()
    original_predictions.tolist()
    return original_labels, original_predictions


def create_data8(datas, pre_len=60, s_len=120):
    values = []
    labels = []

    # 选择你想要的列
    value_columns = [
        'Time',
        'entrance_SO2_concentration',
        'dry_standard_flow_rate',
        'bed_pressure_drop',
        'entrance_gas_temperature',
        'exit_gas_temperature_mean',
        'lime_feeder_1_frequency',
        'lime_feeder_2_frequency',
        'exit_SO2_control_set_value'
    ]
    label_columns = [
        'Time',
        'bag_filter_exit_SO2_concentration'
    ]

    # 使用 DataFrame 的 loc 方法来选择行和列
    value = datas.loc[0:0 + s_len - 1, value_columns].values
    label = datas.loc[:s_len - 1, label_columns].values

    values.append(value)
    labels.append(label)

    return values, labels  # 假设你想要返回 values 和 labels


def read_data8(datalist):
    global first
    feature_names = [
        'Time',
        'entrance_SO2_concentration',
        'dry_standard_flow_rate',
        'bed_pressure_drop',
        'entrance_gas_temperature',
        'exit_gas_temperature_mean',
        'lime_feeder_1_frequency',
        'lime_feeder_2_frequency',
        'bag_filter_exit_SO2_concentration',
        'exit_SO2_control_set_value'
    ]  # 与read_test_data中的特征名称相同
    dataframes = []

    for data in datalist:
        data_values = [data[feature_name] for feature_name in feature_names]
        datas = pd.DataFrame([data_values], columns=feature_names)  # 创建一个DataFrame来模拟CSV文件的结构
        datas.fillna(0, inplace=True)  # 填充NaN值
        dataframes.append(datas)
    # 使用pd.concat()将所有DataFrame拼接起来
    concatenated_data = pd.concat(dataframes, ignore_index=True)
    # 获取concatenated_data[40015]列前60个值
    temp_labels = concatenated_data['bag_filter_exit_SO2_concentration'].values[pre_len:]
    concatenated_data['bag_filter_exit_SO2_concentration'].values[:pre_len] = temp_labels
    # print('\n***************\ndata:', concatenated_data)
    # concatenated_data.to_csv('test.csv')
    values, labels = create_data8(concatenated_data, pre_len=pre_len, s_len=s_len)
    values = np.array(values)
    labels = np.array(labels)
    return values, labels


class Data(BaseModel):
    data: List[List[float]]

    class Config:
        schema_extra = {
            "example": {
                "data": [
                    [i * 0.1 for i in range(60)],
                    [i * 0.2 for i in range(60)]
                ]
            }
        }


class DesulfurizationInput(BaseModel):
    entrance_SO2_concentration: float = Field(..., description="脱硫岛入口烟气SO2折算浓度", example=500.0)
    dry_standard_flow_rate: float = Field(..., description="脱硫岛入口烟气干标流量", example=3000.0)
    bed_pressure_drop: float = Field(..., description="吸收塔床层压降", example=50.0)
    entrance_gas_temperature: float = Field(..., description="吸收塔入口烟气温度", example=150.0)
    exit_gas_temperature_mean: float = Field(..., description="吸收塔出口烟气温度均值", example=120.0)
    lime_feeder_1_frequency: float = Field(..., description="消石灰调频旋转给料器1频率", example=60.0)
    lime_feeder_2_frequency: float = Field(..., description="消石灰调频旋转给料器2频率", example=60.0)
    bag_filter_exit_SO2_concentration: float = Field(..., description="布袋出口烟气SO2折算浓度", example=200.0)
    exit_SO2_control_set_value: float = Field(..., description="出口SO2控制设定值", example=100.0)


@app.post("/predict_SO2/")
async def predict_SO2(input_data: DesulfurizationInput):
    global data_list
    global history_predictions, send_data, predictions, true_values
    input_dict = input_data.dict()
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    time_input_dict = {'Time': current_time}
    print(f'Time: {current_time}')
    time_input_dict.update(input_dict)
    data_list.append(time_input_dict)
    return_value = None
    if len(data_list) > 120:
        data_list.pop(0)  # 从队列左侧删除旧数据
    if len(data_list) == 120:
        print(f'len(data_list):{len(data_list)}')
        # 还要改前面47行左右的model和开头的x、y_stand
        test_x, test_y = read_data8(data_list)
        # 调用推理函数，逐步生成预测结果
        original_labels, original_predictions = inference_streaming(model_8, test_x, test_y, batch_size=1)
        return_value = original_predictions[-1]
        # print(f'original_labels:{original_labels[pre_len:]}')
        # print(f'original_predictions:{original_predictions}')
        predictions_generator = [original_labels[pre_len:].tolist(), original_predictions.tolist()]
        # print(f'predictions_generator:{predictions_generator}')
        history_predictions.pop(0)
        history_predictions.append(predictions_generator[1][-1])
        true_values[0:60] = predictions_generator[0][:]
        predictions[60:] = predictions_generator[1][:]
        send_data = [true_values, predictions, true_values, history_predictions]
        # 新数据可用时，向所有连接的WebSocket发送数据

        url = 'http://47.253.56.195:7860/data/'
        # url = 'http://127.0.0.1:7860/data/'
        # 将您想要发送的数据作为一个字典传递
        # 使用 httpx 异步发送 POST 请求
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(url, json={"data": predictions_generator})
        # 检查响应
        if response.status_code == 200:
            print(f'Successful POST request. Response: {response.json()}')
        else:
            print(f'Failed POST request. Status code: {response.status_code}')

        # for websocket in connected_websockets:
        #     logger.info(websocket)
        #     await websocket.send_json(send_data)
    # logger.info(f'data_list:{data_list}')
    # print(f'data_list:{data_list}')

    return {"status": "data received",
            "last_second_data": return_value}


@app.get("/")
async def read_root():
    # 返回主页面
    return templates.TemplateResponse("index.html", {"request": {}})


@app.post("/data/")
async def receive_data(item: Data):
    global history_predictions, send_data, predictions, true_values
    # 更新历史预测数据
    print(f'Data:{item}')
    history_predictions.pop(0)
    history_predictions.append(item.data[1][-1])
    true_values[0:60] = item.data[0][:]
    predictions[60:] = item.data[1][:]
    send_data = [true_values, predictions, true_values, history_predictions]
    # 新数据可用时，向所有连接的WebSocket发送数据
    print(f'send_data:{send_data}')
    for websocket in connected_websockets:
        logger.info(websocket)
        await websocket.send_json(send_data)
    return {"status": "data received"}


@app.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # 将新连接添加到全局集合
    connected_websockets.add(websocket)
    try:
        while True:
            await asyncio.sleep(1)  # 保持连接打开，直到客户端断开连接
    finally:
        # 在连接关闭时从全局集合中删除
        connected_websockets.remove(websocket)


@app.middleware("http")
async def catch_422_errors(request: Request, call_next):
    response = await call_next(request)
    if response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY:
        body = await response.body()
        return JSONResponse(content={"error": "Validation error", "body": body.decode("utf-8")}, status_code=422)
    return response


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
