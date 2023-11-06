import struct
import time
import pandas as pd
import requests
import threading

url = "https://u137836-8050-9419a366.neimeng.seetacloud.com:6443/predict_SO2/"  # 替换为你的 FastAPI 应用的 URL 和端口
# url = "http://localhost:7860/predict_SO2/"
filename = '../datas/val/table_1.csv'
df = pd.read_csv(filepath_or_buffer=filename, skiprows=range(1, 181))


def send_request(data):
    response = requests.post(url, json=data)
    print(response.status_code)
    print(response)  # 如果你的路由返回 JSON 数据


# 删除不需要的列
feature_names = ['序号', "Unnamed: 0", 'Time']
for feature_name in feature_names:
    df.pop(str(feature_name))

# 创建一个字典，将列名映射到其对应的英文名称
chinese_to_english_mapping = {
    '脱硫岛入口烟气SO2折算浓度': 'entrance_SO2_concentration',
    '脱硫岛入口烟气干标流量': 'dry_standard_flow_rate',
    '吸收塔床层压降': 'bed_pressure_drop',
    '吸收塔入口烟气温度': 'entrance_gas_temperature',
    '吸收塔出口烟气温度': 'exit_gas_temperature_mean',
    '消石灰调频旋转给料器1频率': 'lime_feeder_1_frequency',
    '消石灰调频旋转给料器2频率': 'lime_feeder_2_frequency',
    '布袋出口烟气SO2折算浓度': 'bag_filter_exit_SO2_concentration',
    '出口SO2控制设定值': 'exit_SO2_control_set_value'
}

# 使用rename方法将列名从中文更改为英文
df.rename(columns=chinese_to_english_mapping, inplace=True)

# 遍历数据帧的每一行
for index, row in df.iterrows():
    # 遍历每一列
    # time.sleep(1)
    sended_data = {}
    for column, value in row.iteritems():
        # 此时的 column 变量将包含英文列名
        sended_data.update({column: value})
    print(sended_data)
    # 创建并启动一个新线程来发送请求
    threading.Thread(target=send_request, args=(sended_data,)).start()

    # 主线程将等待1秒，然后继续处理下一行数据
    time.sleep(1)
    # response = requests.post(url, json=sended_data)
    # print(response.status_code)
    # print(response)  # 如果你的路由返回 JSON 数据
