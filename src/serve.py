from fastapi import FastAPI, HTTPException, Request
import torch
from pydantic import BaseModel

from launch import launch_multi_model


app = FastAPI()

@app.post("/test/{model_id}")
async def test(request: Request):
    pass
