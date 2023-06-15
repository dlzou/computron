from fastapi import FastAPI, HTTPException, Request
import torch
from pydantic import BaseModel

from computron import launch_computron


app = FastAPI()


@app.post("/test/{model_id}")
async def test(request: Request):
    pass
