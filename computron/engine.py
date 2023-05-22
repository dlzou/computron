import asyncio
from collections import deque
from collections.abc import Hashable
from enum import Enum
import signal
import time

from colossalai.logging import get_dist_logger
from energonai.pipe import Pipe
from energonai.utils import build_device_maps, Terminator
import torch.distributed.rpc as trpc

from computron.batch_manager import BatchManager
from computron.config import EngineConfig, ModelConfig
from computron.messages import (
    LoadEntry,
    SubmitEntry,
    TaskEntry,
)
from computron.utils import send_obj, recv_obj


class SubmitQueueFull(Exception):
    pass


class LoadState(Enum):
    OFFLOADED = 0
    LOADED = 1
    LOADING = 2


class EntryCounter:
    """Count entry responses from workers to know when it fully completes."""
    def __init__(self, wait_count: int):
        assert wait_count > 0, "must wait for at least one entry response"
        self.entry = None
        self.counter = 0
        self.wait_count = wait_count
    
    def increment(self, entry: TaskEntry | LoadEntry):
        self.counter += 1
        if self.counter == 1:
            self.entry = entry
        # TODO: validate all entries are the same
        # elif self.entry != entry:
        #     raise RuntimeError("entries do not match")
        if self.counter == self.wait_count:
            # Received entries from all necessary workers
            return True
        return False


class Engine:
    """
    Adapted from https://github.com/hpcaitech/EnergonAI/blob/main/energonai/engine.py
    with significant chagnes.
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        model_configs: list[ModelConfig],
        tp_world_size: int,
        pp_world_size: int,
        n_proc_per_node: int,
    ):
        self.tp_world_size = tp_world_size
        self.pp_world_size = pp_world_size
        self.world_size = tp_world_size * pp_world_size
        rpc_options = {}
        if engine_config.rpc_disable_shm:
            # SHM may lead to timeout error. Disabling SHM and only enabling uv transport can solve this problem.
            # See https://discuss.pytorch.org/t/rpc-behavior-difference-between-pytorch-1-7-0-vs-1-9-0/124772/5
            # This is a workaround and may be solved in the future.
            rpc_options["_transports"] = ["uv"]
        trpc.init_rpc(
            "master",
            rank=0,
            world_size=self.world_size + 1,
            rpc_backend_options=trpc.TensorPipeRpcBackendOptions(
                init_method=f"tcp://{engine_config.master_host}:{engine_config.rpc_port}",
                device_maps=build_device_maps(self.world_size, n_proc_per_node),
                **rpc_options,
            ),
        )
        self.from_worker_pipes: list[Pipe] = []
        for i in range(self.world_size):
            pipe = Pipe(f"{i}_to_m", f"worker{i}", "master")
            self.from_worker_pipes.append(pipe)
        self.submit_pipes: list[Pipe] = []
        # self.completion_pipes: list[Pipe] = []
        for i, pipe in enumerate(self.from_worker_pipes):
            worker_pp_rank = pipe.recv()
            if worker_pp_rank == 0:
                self.submit_pipes.append(
                    Pipe(f"m_to_{i}", "master", f"worker{i}", max_size=engine_config.pipe_size)
                )
            # if worker_pp_rank == pp_world_size - 1:
            #     self.completion_pipes.append(pipe)

        self.batch_managers: dict[str, BatchManager] = {}
        self.queue_size = engine_config.queue_size
        self.submit_queues: dict[str, deque[SubmitEntry]] = {}
        # self.can_submit: dict[str, bool] = {}
        self.submit_locks: dict[str, asyncio.Lock] = {}
        self.loaded: dict[str, LoadState] = {}
        for mc in model_configs:
            m = mc.model_id
            if mc.batch_manager is None:
                self.batch_managers[m] = BatchManager()
            else:
                assert isinstance(mc.batch_manager, BatchManager)
                self.batch_managers[m] = mc.batch_manager
            self.submit_queues[m] = deque()
            # self.submit_locks[m] = asyncio.Lock()
            self.loaded[m] = LoadState.LOADED # LoadState.OFFLOADED

        self.completion_map: dict[Hashable, object] = {}
        self.completion_events: dict[Hashable, asyncio.Event] = {}

        self.batch_info: dict[Hashable, object] = {}
        self.timer_info: dict[Hashable, tuple[int, float]] = {}
        self.entry_counters: dict[Hashable, EntryCounter] = {}
        self.tasks = []
        self.loop = None

        self.evict_queue: list[str] = []
        self.max_loaded = engine_config.max_loaded

        self.logger = get_dist_logger("computron")
        self.logger.info(f"engine started")
        # self._start()

    async def run(self):
        # loop = asyncio.new_event_loop()
        # shutdown_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        # for s in shutdown_signals:
        #     loop.add_signal_handler(s, lambda: loop.create_task(self.shutdown(loop)))
        # # TODO: add exception handler
        # try:
        #     loop.create_task(self._request_server())
        #     loop.create_task(self._submit_loop())
        #     loop.create_task(self._completion_loop())
        #     loop.run_forever()
        # finally:
        #     loop.close()

        self.loop = asyncio.get_running_loop()
        shutdown_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        for s in shutdown_signals:
            self.loop.add_signal_handler(s, lambda: self.loop.create_task(self.shutdown(True)))
        
        submit_task = self.loop.create_task(self._submit_loop())
        completion_task = self.loop.create_task(self._completion_loop())
        self.tasks = [submit_task, completion_task]
        await asyncio.gather(*self.tasks)

    async def shutdown(self, stop_loop=False):
        Terminator.shield()
        for i in range(self.world_size):
            trpc.rpc_sync(f"worker{i}", Terminator.terminate)
        trpc.shutdown()

        if self.loop is not None:
            for t in self.tasks:
                t.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            if stop_loop:
                asyncio.get_running_loop().stop()

    async def submit(self, model_id: str, data: object):
        uid = hash((id(data), time.time()))
        assert uid not in self.completion_map
        entry = SubmitEntry(uid, model_id, data)
        if self.queue_size > 0 and len(self.submit_queues[model_id]) >= self.queue_size:
            raise SubmitQueueFull(f"{model_id} submit queue full")
        self.completion_events[entry.uid] = asyncio.Event()
        # async with self.submit_locks[model_id]:
        self.submit_queues[model_id].append(entry)
        # Sleep until request completed
        await self.completion_events[uid].wait()
        del self.completion_events[uid]
        return self.completion_map.pop(uid)

    # async def handle_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    #     req = await recv_obj(reader)
    #     if isinstance(req, self.request_type):
    #         entry: SubmitEntry = self.unpack_request_fn(req)
    #         assert entry.uid not in self.completion_map
    #         if self.queue_size > 0 and len(self.submit_queue) >= self.queue_size:
    #             raise SubmitQueueFull(f"Submit queue full, size: {self.queue_size}")
    #         self.completion_event[entry.uid] = asyncio.Event()
    #         self.submit_queue.append(entry)
    #         # Sleep until completion signal
    #         await self.completion_event[entry.uid].wait()
    #         del self.completion_event[entry.uid]
    #         output = self.completion_map.pop(entry.uid)
    #         resp = self.pack_response_fn(output)
    #         await send_obj(writer, resp)
    #         writer.close()
    #         await writer.wait_closed()

    #     elif isinstance(req, LoadRequest):
    #         entry: LoadEntry = LoadEntry(id(req), req.load, req.blocking)
    #         async with self.submit_cv:
    #             # if self.can_submit != entry.load:
    #             for pipe in self.submit_pipes:
    #                 pipe.send(entry)
    #             for pipe in self.from_worker_pipes:
    #                 pipe.recv()
    #             self.can_submit = entry.load
    #             if self.can_submit:
    #                 self.submit_cv.notify()
    #        # if not entry.blocking:
    #         #     await send_obj(writer, LoadResponse(success=True))
    #         #     writer.close()
    #         #     await writer.wait_closed()
    #         self.logger.info(f"{self.model_id} loaded state: {entry.load}")
    #         # if entry.blocking:
    #         await send_obj(writer, LoadResponse(success=True))
    #         writer.close()
    #         await writer.wait_closed()

    #     elif isinstance(req, PingRequest):
    #         await send_obj(writer, PingResponse())
    #         writer.close()
    #         await writer.wait_closed()
        
    #     else:
    #         raise RuntimeError(f"Unknown request type: {type(req)}")

    # async def _request_server(self):
    #     server = await asyncio.start_server(
    #         lambda r, w: self.handle_request(r, w),
    #         self.master_host,
    #         self.request_port,
    #     )
    #     async with server:
    #         await server.serve_forever()

    async def _swap_in(self, model_id: str):
        if len(self.evict_queue) >= self.max_loaded:
            out_model_id = self.evict_queue.pop(0)
            out_uid = hash((out_model_id, time.time()))
            out_entry = LoadEntry(out_uid, out_model_id, False, False)
            for pipe in self.submit_pipes:
                pipe.send(out_entry)
            self.loaded[out_model_id] = False
        uid = hash((model_id, time.time()))
        entry = LoadEntry(uid, model_id, True, False)
        self.timer_info[entry.uid] = (0, time.time())
        self.entry_counters[entry.uid] = EntryCounter(self.world_size)
        for pipe in self.submit_pipes:
            pipe.send(entry)
        self.loaded[model_id] = True
        self.evict_queue.append(model_id)

    async def _submit_loop(self):
        while True:
            for model_id, submit_queue in self.submit_queues.items():
                # async with self.submit_locks[model_id]:
                if len(submit_queue) > 0: # and self.loaded[model_id]:
                    if not self.loaded[model_id]:
                        await self._swap_in(model_id)
                    entry, batch_info = self.batch_managers[model_id].make_batch(submit_queue)
                    self.batch_info[entry.uids] = batch_info
                    self.timer_info[entry.uids] = (len(entry.uids), time.time())
                    self.entry_counters[entry.uids] = EntryCounter(self.tp_world_size)
                    for pipe in self.submit_pipes:
                        pipe.send(entry)
            await asyncio.sleep(0)

    async def _completion_loop(self):
        # received_data: dict[int, object] = {}
        while True:
            for pipe in self.from_worker_pipes:
                try:
                    entry = pipe.recv_nowait()
                    if isinstance(entry, TaskEntry) and self.entry_counters[entry.uids].increment(entry):
                        batch_info = self.batch_info.pop(entry.uids)
                        for uid, output in self.batch_managers[entry.model_id].split_batch(entry, **batch_info):
                            self.completion_map[uid] = output
                            self.completion_events[uid].set()
                        batch_size, start_time = self.timer_info.pop(entry.uids)
                        self.logger.info(
                            f"{entry.model_id} batch size: {batch_size}, time: {time.time() - start_time:.3f}"
                        )
                    elif isinstance(entry, LoadEntry) and self.entry_counters[entry.uid].increment(entry):
                        pass
                except RuntimeError:
                    pass

            # for i, pipe in enumerate(self.completion_pipes):
            #     if i not in received_data:
            #         try:
            #             received_data[i] = pipe.recv_nowait()
            #         except RuntimeError:
            #             pass
            # if len(received_data) == len(self.completion_pipes):
            #     # TODO: validate all entries are the same
            #     entries: list[TaskEntry | LoadEntry] = list(
            #         map(
            #             lambda k: received_data[k],
            #             sorted(received_data.keys()),
            #         )
            #     )
            #     received_data.clear()
            #     entry0 = entries[0]
            #     if isinstance(entry0, TaskEntry):
            #         batch_info = self.batch_info.pop(entry0.uids)
            #         for uid, output in self.batch_managers[entry0.model_id].split_batch(entry0, **batch_info):
            #             self.completion_map[uid] = output
            #             self.completion_events[uid].set()
            #         batch_size, start_time = self.timer_info.pop(entry0.uids)
            #         self.logger.info(
            #             f"{entry0.model_id} batch size: {batch_size}, time: {time.time() - start_time:.3f}"
            #         )
            #     elif isinstance(entry0, LoadEntry):
            #         self.completion_events[entry0.uid].set()
            await asyncio.sleep(0)
