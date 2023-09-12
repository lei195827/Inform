import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
from fastapi import BackgroundTasks

app = FastAPI()

app.mount("/node_modules", StaticFiles(directory="node_modules"), name="node_modules")
templates = Jinja2Templates(directory="templates")


class Data(BaseModel):
    data: list


data_queue = []


@app.get("/")
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})


@app.post("/data/")
async def receive_data(item: Data):
    # print(Data)
    data_queue.append(item.data)
    # print(data_queue)
    return {"status": "data received"}


@app.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Connected")
    while True:
        if data_queue:
            data = data_queue.pop(0)
            await websocket.send_json(data)
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=7860, reload=True)
