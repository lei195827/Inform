import asyncio
import datetime
import logging
from typing import List
import uvicorn
from fastapi import FastAPI, WebSocket, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import websockets.exceptions

# 初始化FastAPI应用
app = FastAPI()
connected_websockets = set()
# 挂载静态文件夹
app.mount("/node_modules", StaticFiles(directory="node_modules"), name="node_modules")

# 设置模板目录
templates = Jinja2Templates(directory="templates")

# 初始化数据队列和其他相关数据
data_queue = []
history_predictions = [None] * 120
predictions = [None] * 60
predictions.extend(np.zeros(60).tolist())
true_values = np.zeros(60).tolist()
true_values.extend([None] * 60)
send_data = np.zeros((4, 120)).tolist()


class Data(BaseModel):
    data: List[List[float]]

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    [i * 0.1 for i in range(60)],
                    [i * 0.2 for i in range(60)]
                ]
            }
        }


@app.get("/")
async def read_root():
    # 返回主页面
    return templates.TemplateResponse("index.html", {"request": {}})


@app.post("/data/")
async def receive_data(item: Data):
    global history_predictions, send_data, predictions, true_values
    # 更新历史预测数据
    history_predictions.pop(0)
    history_predictions.append(item.data[1][-1])
    true_values[0:60] = item.data[0][:]
    predictions[60:] = item.data[1][:]
    send_data = [true_values, predictions, true_values, history_predictions]
    # 新数据可用时，向所有连接的WebSocket发送数据
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f'current_time:{current_time}')

    closed_websockets = set()  # 用于存储已关闭的WebSocket连接
    for websocket in connected_websockets:
        try:
            await websocket.send_json(send_data)  # 尝试发送数据
        except websockets.exceptions.ConnectionClosedOK:
            closed_websockets.add(websocket)  # 如果连接已关闭，添加到closed_websockets集合中
        except Exception as e:  # 捕获并处理其他可能的异常
            logging.error(f"Error sending message to websocket: {e}")
            closed_websockets.add(websocket)

    # 从connected_websockets集合中移除已关闭的WebSocket连接
    connected_websockets.difference_update(closed_websockets)

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


if __name__ == "__main__":
    uvicorn.run("app_so2:app", host="0.0.0.0", port=7860, reload=True)
