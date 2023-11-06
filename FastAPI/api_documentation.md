
### API 文档

---

#### API 名称: 提交环境数据

#### 请求URL: 
```
http://region-9.autodl.pro:36605/predict_SO2/
```

#### 请求方法: 
POST

#### 请求参数:

| 参数名                                   | 类型     | 描述                           | 是否必需 | 示例值   |
|--------------------------------------|--------|------------------------------|--------|--------|
| entrance_SO2_concentration           | 浮点数  | 脱硫岛入口烟气SO2折算浓度        | 是      | 500.0  |
| dry_standard_flow_rate               | 浮点数  | 脱硫岛入口烟气干标流量           | 是      | 3000.0 |
| bed_pressure_drop                    | 浮点数  | 吸收塔床层压降                 | 是      | 50.0   |
| entrance_gas_temperature             | 浮点数  | 吸收塔入口烟气温度             | 是      | 150.0  |
| exit_gas_temperature_mean            | 浮点数  | 吸收塔出口烟气温度             | 是      | 120.0  |
| lime_feeder_1_frequency              | 浮点数  | 消石灰调频旋转给料器1频率        | 是      | 60.0   |
| lime_feeder_2_frequency              | 浮点数  | 消石灰调频旋转给料器2频率        | 是      | 60.0   |
| bag_filter_exit_SO2_concentration   | 浮点数  | 布袋出口烟气SO2折算浓度         | 是      | 200.0  |
| exit_SO2_control_set_value           | 浮点数  | 出口SO2控制设定值              | 是      | 100.0  |

#### 请求示例:

```json
{
  "entrance_SO2_concentration": 500.0,
  "dry_standard_flow_rate": 3000.0,
  "bed_pressure_drop": 50.0,
  "entrance_gas_temperature": 150.0,
  "exit_gas_temperature_mean": 120.0,
  "lime_feeder_1_frequency": 60.0,
  "lime_feeder_2_frequency": 60.0,
  "bag_filter_exit_SO2_concentration": 200.0,
  "exit_SO2_control_set_value": 100.0
}
```

#### 响应格式:

响应数据将是一个JSON对象，包含操作的状态和消息。

示例：

```json
{
  "status": "success",
  "message": "data received."
}
```

---
