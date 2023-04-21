import asyncio
import pickle
from typing import Any, Dict, Tuple

from energonai import BatchManager
from pydantic import BaseModel

from engine import OffloadingEngine
from utils import send_obj, recv_obj


class Controller:

    def __init__(self):
        self.models: Dict[str, Tuple[str, int]] = {}
        # TODO: state for offloading algorithm

    def register_model(self, model_id: str, host: str, port: int):
        assert model_id not in self.models, f"'{model_id}' already registered"
        self.models[model_id] = (host, port)

    async def handle_request(self, model_id: str, req: BaseModel):
        # TODO: computations for offloading
        reader, writer = await asyncio.open_connection(*self.models[model_id])
        await send_obj(writer, req)
        resp = await recv_obj(reader)
        writer.close()
        await writer.wait_closed()
        return resp
