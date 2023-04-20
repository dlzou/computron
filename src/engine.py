import asyncio
from collections import deque
from dataclasses import dataclass
import signal
import time
from typing import Any, Callable, Deque, Dict, Hashable, List, Optional, Tuple

from colossalai.logging import get_dist_logger
from energonai import BatchManager, SubmitEntry, TaskEntry
from energonai.pipe import Pipe
from energonai.utils import Terminator, build_device_maps, use_lock
from pydantic import BaseModel
import torch.distributed.rpc as trpc

from utils import send_obj, recv_obj


@dataclass
class OffloadEntry:
    uid: Hashable
    on_device: bool


class OffloadingEngine:
    """
    Adapted from https://github.com/hpcaitech/EnergonAI/blob/main/energonai/engine.py
    with significant chagnes.
    """

    def __init__(
        self,
        tp_world_size: int,
        pp_world_size: int, 
        master_host: str,
        rpc_port: int,
        request_port: int,
        request_type: BaseModel,
        unpack_request_fn: Callable,
        pack_response_fn: Callable,
        n_proc_per_node: int,
        batch_manager: Optional[BatchManager] = None,
        pipe_size: int = 1,
        queue_size: int = 0,
        rpc_disable_shm: bool = True,
    ):
        self.logger = get_dist_logger('energonai')
        if batch_manager is None:
            self.batch_manager = BatchManager()
        else:
            assert isinstance(batch_manager, BatchManager)
            self.batch_manager = batch_manager
        self.world_size = tp_world_size * pp_world_size

        rpc_options = {}
        if rpc_disable_shm:
            # SHM may lead to timeout error. Disabling SHM and only enabling uv transport can solve this problem.
            # See https://discuss.pytorch.org/t/rpc-behavior-difference-between-pytorch-1-7-0-vs-1-9-0/124772/5
            # This is a workaround and may be solved in the future.
            rpc_options['_transports'] = ['uv']
        trpc.init_rpc('master', rank=0, world_size=self.world_size + 1,
                      rpc_backend_options=trpc.TensorPipeRpcBackendOptions(
                          init_method=f'tcp://{master_host}:{rpc_port}',
                          device_maps=build_device_maps(self.world_size, n_proc_per_node),
                          **rpc_options
                      ))
        self.from_worker_pipes: List[Pipe] = []
        for i in range(self.world_size):
            pipe = Pipe(f'{i}_to_m', f'worker{i}', 'master')
            self.from_worker_pipes.append(pipe)
        self.submit_pipes: List[Pipe] = []
        self.completion_pipes: List[Pipe] = []
        for i, pipe in enumerate(self.from_worker_pipes):
            worker_pp_rank = pipe.recv()
            if worker_pp_rank == 0:
                self.submit_pipes.append(
                    Pipe(f'm_to_{i}', 'master', f'worker{i}',max_size=pipe_size)
                )
            if worker_pp_rank == pp_world_size - 1:
                self.completion_pipes.append(pipe)

        assert queue_size > 0, "queue_size must be at least 1"
        self.queue_size = queue_size
        self.submit_queue: Deque[SubmitEntry] = deque()
        self.batch_info: Dict[Hashable, Any] = {}
        self.timer_info: Dict[Hashable, Tuple[int, float]] = {}
        self.completion_map: Dict[Hashable, Any] = {}
        self.completion_event: Dict[Hashable, asyncio.Event] = {}
        
        self.master_host = master_host
        self.request_port = request_port
        self.request_type: BaseModel = request_type
        self.unpack_request_fn: Callable = unpack_request_fn
        self.pack_response_fn: Callable = pack_response_fn

        self.logger.info('Engine start')
        self._start()


    def _start(self):
        loop = asyncio.new_event_loop()
        shutdown_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        for s in shutdown_signals:
            loop.add_signal_handler(s, lambda: loop.create_task(self._shutdown(loop)))
        try:
            loop.create_task(self._request_server())
            loop.create_task(self._submit_loop())
            loop.create_task(self._completion_loop())
            loop.run_forever()
        finally:
            loop.close()


    async def _shutdown(self, loop):
        # TODO: shutdown workers
        Terminator.shield()
        for i in range(self.world_size):
            trpc.rpc_sync(f'worker{i}', Terminator.terminate)
        trpc.shutdown()

        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()


    async def _handle_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        req = await recv_obj(reader)
        if isinstance(req, self.request_type):
            entry: SubmitEntry = self.unpack_request_fn(req)
            self.completion_event[entry.uid] = asyncio.Event()
            self.submit_queue.append(entry)
            await self.completion_event[entry.uid].wait()

            del self.completion_event[entry.uid]
            output = self.completion_map.pop(entry.uid)
            resp = self.pack_response_fn(output)
            await send_obj(writer, resp)
        
        # TODO: handle OffloadRequest

        writer.close()
        await writer.wait_closed()


    async def _request_server(self):
        server = await asyncio.start_server(
            lambda r, w: self._handle_request(r, w),
            self.master_host,
            self.request_port,
        )
        async with server:
            await server.serve_forever()


    async def _submit_loop(self):
        while True:
            if len(self.submit_queue) > 0:
                # TODO: handle OffloadEntry
                task_entry, batch_info = self.batch_manager.make_batch(self.submit_queue)
                self.batch_info[task_entry.uids] = batch_info
                self.timer_info[task_entry.uids] = (len(task_entry.uids), time.time())
                for pipe in self.submit_pipes:
                    pipe.send(task_entry)
            else:
                await asyncio.sleep(0.01)
    

    async def _completion_loop(self):
        received_data: Dict[int, Any] = {}
        while True:
            for i, pipe in enumerate(self.completion_pipes):
                if i not in received_data:
                    try:
                        received_data[i] = pipe.recv_nowait()
                    except RuntimeError:
                        pass
            if len(received_data) == len(self.completion_pipes):
                # TODO: validate they are all the same
                task_entries: List[TaskEntry] = list(map(lambda k: received_data[k], sorted(received_data.keys())))
                received_data.clear()
                batch_info = self.batch_info.pop(task_entries[0].uids)
                for uid, output in self.batch_manager.split_batch(task_entries[0], **batch_info):
                    self.completion_map[uid] = output
                batch_size, start_time = self.timer_info.pop(task_entries[0].uids)
                self.logger.info(f'batch size: {batch_size}, time: {time.time() -start_time:.3f}')
            else:
                await asyncio.sleep(0.01)
