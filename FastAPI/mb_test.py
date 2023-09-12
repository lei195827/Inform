from pymodbus.server.sync import StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
from pymodbus.datastore import ModbusSparseDataBlock

# 初始化一个最大长度为120的双端队列
data_list = []


class CustomDataBlock(ModbusSparseDataBlock):
    def __init__(self, values):
        super().__init__(values)
        self.i = 0
        self.temp_values = {}  # 用于跟踪已更新的地址和它们的值

    def setValues(self, address, values):
        # 更新临时值
        self.temp_values[address] = values[0]  # 假设每次只更新一个值
        # 检查是否所有的地址都已更新
        if set(self.temp_values.keys()) == set(initial_values.keys()):
            # 将整组数据作为一个字典存储在data_list中
            data_list.append(self.temp_values.copy())
            self.temp_values.clear()  # 清空临时值

            if len(data_list) > 120:
                self.i = self.i + 1
                print(f"{self.i} i")
                data_list.pop(0)  # 从队列左侧删除旧数据

        super().setValues(address, values)


# 根据您的需求，初始化数据块
initial_values = {
    40001: 0,  # 脱硫岛入口烟气SO2折算浓度
    40003: 0,  # 脱硫岛入口烟气干标流量
    40005: 0,  # 吸收塔床层压降
    40007: 0,  # 吸收塔入口烟气温度
    40009: 0,  # 吸收塔出口烟气温度均值
    40011: 0,  # 消石灰调频旋转给料器1频率
    40013: 0,  # 消石灰调频旋转给料器2频率
    40015: 0,  # 布袋出口烟气SO2折算浓度
    40017: 0,  # 出口SO2控制设定值
    40019: 0  # 出口SO2模型预测值
}

# 使用自定义的数据块初始化保持寄存器
store = ModbusSlaveContext(
    hr=CustomDataBlock(initial_values)
)

context = ModbusServerContext(slaves=store, single=True)
print('server start')
# 启动服务器
StartTcpServer(context, address=('0.0.0.0', 502))
