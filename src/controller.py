import asyncio
import pickle
from typing import Any

from energonai import BatchManager

from engine import OffloadingEngine
from utils import send_obj, recv_obj

class Controller:
    def __init__(self):
        self.model_readers = {}
        self.model_writers = {}
        
    async def register_model(self, model_id: str, host: str, port: int):
        reader, writer = await asyncio.open_connection(host, port)
        self.model_readers[model_id] = reader
        self.model_writers[model_id] = writer

    async def handle_request(self, model_id: str, request_obj: Any):
        await send_obj(self.model_writers[model_id], request_obj)
        result = await recv_obj(self.model_readers[model_id])
        print(result)
