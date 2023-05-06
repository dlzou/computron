from abc import ABC, abstractmethod
import asyncio
from typing import Dict, List, Tuple

from pydantic import BaseModel

from computron.messages import PingRequest, LoadRequest, LoadResponse
from computron.utils import send_obj, recv_obj


"""
Ideas for more controllers with more sophisticated offloading and scheduling strategies:
- Cap total GPU memory used instead of number of models.
- Consider dependencies when scheduling requests, such as for autoregressive generation.
- Prefetch models into GPU memory.
"""


class Controller:
    """Dispatch requests to the target model, performing offloading as needed."""

    @abstractmethod
    def register_model(self, model_id: str, host: str, port: int):
        raise NotImplementedError()

    @abstractmethod
    async def handle_request(self, model_id: str, req: BaseModel):
        raise NotImplementedError()


class LRUController(Controller):
    """Controller with simple LRU policy to cap the number of loaded models."""

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
        asyncio.run(self._wait_model_init(model_id))

    async def _wait_model_init(self, model_id):
        while True:
            try:
                reader, writer = await asyncio.open_connection(*self.engines[model_id])
                await send_obj(writer, PingRequest())
                await recv_obj(reader)
                writer.close()
                await writer.wait_closed()
                break
            except OSError:
                await asyncio.sleep(0.1)

    async def get_heartbeat(self, model_id):
        fut = asyncio.open_connection(*self.engines[model_id])
        try:
            reader, writer = await asyncio.wait_for(fut, timeout=5)
            await send_obj(writer, PingRequest())
            await recv_obj(reader)
            return True
        except asyncio.TimeoutError:
            return False

    async def _swap_in(self, in_model_id: str):
        assert not self.loaded[in_model_id], f"{in_model_id} already loaded"
        swap_out = len(self.evict_queue) >= self.max_loaded
        if swap_out:
            out_model_id = self.evict_queue.pop(0)
            out_reader, out_writer = await asyncio.open_connection(*self.engines[out_model_id])
            load_req = LoadRequest(load=False, flush=True)
            await send_obj(out_writer, load_req)
            load_resp: LoadResponse = await recv_obj(out_reader)
            assert load_resp.success
            self.loaded[out_model_id] = False
            out_writer.close()

        in_reader, in_writer = await asyncio.open_connection(*self.engines[in_model_id])
        load_req = LoadRequest(load=True, flush=False)
        await send_obj(in_writer, load_req)
        load_resp: LoadResponse = await recv_obj(in_reader)
        assert load_resp.success
        in_writer.close()

        self.loaded[in_model_id] = True
        self.evict_queue.append(in_model_id)
        if swap_out:
            await out_writer.wait_closed()
        await in_writer.wait_closed()

    async def handle_request(self, model_id: str, req: BaseModel):
        async with self.request_lock:
            if not self.loaded[model_id]:
                await self._swap_in(model_id)
            else:
                self.evict_queue.remove(model_id)
                self.evict_queue.append(model_id)
            reader, writer = await asyncio.open_connection(*self.engines[model_id])
            await send_obj(writer, req)

        resp = await recv_obj(reader)
        writer.close()
        await writer.wait_closed()
        return resp
