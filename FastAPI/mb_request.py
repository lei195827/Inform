from pymodbus.client.sync import ModbusTcpClient
# 创建客户端实例
client = ModbusTcpClient('127.0.0.1', port=502)

# 连接到服务器
client.connect()
print(client)
for i in range(130):
    for j in range(10):
        client.write_register(40000 + 2*j, i)
# 断开连接
client.close()
