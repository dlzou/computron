import asyncio
from typing import Dict, List, Tuple

from pydantic import BaseModel

from offload import OffloadRequest, OffloadResponse
from utils import send_obj, recv_obj


class Controller:
    def __init__(self, max_loaded: int = 1):
        self.engines: Dict[str, Tuple[str, int]] = {}
        self.loaded: Dict[str, bool] = {}
        self.request_lock = asyncio.Lock()
        self.evict_queue: List[str] = []
        self.max_loaded = max_loaded 

    def register_model(self, model_id: str, host: str, port: int):
        assert model_id not in self.engines, f"'{model_id}' already registered"
        self.engines[model_id] = (host, port)
        self.loaded[model_id] = False
        
    async def _swap_in(self, in_model_id: str):
        # TODO: better swapping algorithm for >2 models?
        assert not self.loaded[in_model_id], f"{in_model_id} already loaded"
        swap_out = len(self.evict_queue) >= self.max_loaded
        if swap_out:
            out_model_id = self.evict_queue.pop(0)
            out_reader, out_writer = await asyncio.open_connection(*self.engines[out_model_id])
            load_req = OffloadRequest(loaded=False)
            await send_obj(out_writer, load_req)
        in_reader, in_writer = await asyncio.open_connection(*self.engines[in_model_id])
        load_req = OffloadRequest(loaded=True)
        await send_obj(in_writer, load_req)

        if swap_out:
            load_resp: OffloadResponse = await recv_obj(out_reader)
            assert load_resp.success
            self.loaded[out_model_id] = False
            out_writer.close()
        load_resp: OffloadResponse = await recv_obj(in_reader)
        assert load_resp.success
        self.loaded[in_model_id] = True
        in_writer.close()

        if swap_out:
            await out_writer.wait_closed()
        await in_writer.wait_closed()
        self.evict_queue.append(in_model_id)

    async def handle_request(self, model_id: str, req: BaseModel):
        async with self.request_lock:
            if not self.loaded[model_id]:
                await self._swap_in(model_id)
            reader, writer = await asyncio.open_connection(*self.engines[model_id])
            await send_obj(writer, req)

        resp = await recv_obj(reader)
        writer.close()
        await writer.wait_closed()
        return resp
