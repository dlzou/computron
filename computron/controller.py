import asyncio
import pickle
from typing import Any, Dict, Tuple

from energonai import BatchManager
from pydantic import BaseModel

from engine import OffloadingEngine
from offload import OffloadRequest, OffloadResponse
from utils import send_obj, recv_obj


class Controller:

    def __init__(self):
        self.models: Dict[str, Tuple[str, int]] = {}
        self.loaded: Dict[str, bool] = {}
        self.stream_lock: Dict[str, asyncio.Lock] = {}

    def register_model(self, model_id: str, host: str, port: int):
        assert model_id not in self.models, f"'{model_id}' already registered"
        self.models[model_id] = (host, port)
        self.loaded[model_id] = False
        
    async def _send_load_request(self, model_id: str, loaded: bool):
        reader, writer = await asyncio.open_connection(*self.models[model_id])
        load_req = OffloadRequest(loaded=loaded)
        await send_obj(writer, load_req)
        load_resp: OffloadResponse = await recv_obj(reader)
        assert load_resp.success
        writer.close()
        await writer.wait_closed()
        self.loaded[model_id] = True

    async def handle_request(self, model_id: str, req: BaseModel):
        # TODO: computations for offloading
        if not self.loaded[model_id]:
            await self._send_load_request(model_id, True)

        reader, writer = await asyncio.open_connection(*self.models[model_id])

        await send_obj(writer, req)
        resp = await recv_obj(reader)
        writer.close()
        await writer.wait_closed()
        return resp
