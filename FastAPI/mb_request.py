import struct
import time
from time import sleep

import pandas as pd
from pymodbus.client.sync import ModbusTcpClient


def float_to_two_registers(value):
    """将浮点数转换为两个16位整数"""
    packed_value = struct.pack('>f', value)
    registers = struct.unpack('>HH', packed_value)
    return registers


filename = '../datas/val/table_1.csv'
df = pd.read_csv(filepath_or_buffer=filename, skiprows=range(1, 181))

# 删除不需要的列
feature_names = ['序号', "Unnamed: 0"]
for feature_name in feature_names:
    df.pop(str(feature_name))

# 创建一个字典，将列名映射到其对应的MODBUS地址
address_mapping = {
    "脱硫岛入口烟气SO2折算浓度": 40001,
    "脱硫岛入口烟气干标流量": 40003,
    "吸收塔床层压降": 40005,
    "吸收塔入口烟气温度": 40007,
    "吸收塔出口烟气温度": 40009,
    "消石灰调频旋转给料器1频率": 40011,
    "消石灰调频旋转给料器2频率": 40013,
    "布袋出口烟气SO2折算浓度": 40015,
    "出口SO2控制设定值": 40017
}

# 创建客户端实例
client = ModbusTcpClient('127.0.0.1', port=502)
# 连接到服务器
client.connect()

# 遍历数据帧的每一行
for index, row in df.iterrows():
    # 遍历每一列
    time.sleep(1)
    for column, value in row.iteritems():
        # 获取该列对应的MODBUS地址
        address = address_mapping.get(column)
        if address:
            # 写入数据
            # if address == 40016:
                # print(f'address:{address},value:{value}')
            registers = float_to_two_registers(float(value))
            result = client.write_registers(address - 1, registers)

            # print(f'address:{address},registers:{registers}')
            # print(f'result:{result}')
# 断开连接
client.close()

