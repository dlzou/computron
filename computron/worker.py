from contextlib import contextmanager
from queue import Queue
import threading
import time

import colossalai
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.pipeline.pipelinable import PipelinableContext
from energonai.pipe import Pipe
from energonai.utils import build_device_maps, Terminator
import torch
import torch.distributed.rpc as trpc

from computron.launch import EngineConfig, ModelConfig
from computron.messages import LoadEntry, TaskEntry


class Worker:
    """
    Adapted from https://github.com/hpcaitech/EnergonAI/blob/main/energonai/worker.py
    with significant chagnes.
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        model_configs: list[ModelConfig],
        rank: int,
        tp_world_size: int,
        pp_world_size: int,
        n_proc_per_node: int,
    ):
        self.global_rank = rank
        self.world_size = tp_world_size * pp_world_size
        self.tp_world_size = tp_world_size
        self.pp_world_size = pp_world_size
        disable_existing_loggers(exclude=["computron", "colossalai"])
        colossalai.launch(
            {
                "parallel": {
                    "tensor": {"mode": "1d", "size": tp_world_size},
                    "pipeline": pp_world_size,
                }
            },
            rank,
            self.world_size,
            engine_config.master_host,
            engine_config.master_port,
        )
        self.tp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
        self.pp_rank = (
            gpc.get_local_rank(ParallelMode.PIPELINE)
            if gpc.is_initialized(ParallelMode.PIPELINE)
            else 0
        )
        self.models = {}
        for mc in model_configs:
            m = mc.model_id
            if mc.pipelinable:
                pctx = PipelinableContext()
                with pctx:
                    _ = mc.model_fn(**mc.model_kwargs)
                pctx.to_layer_list()
                pctx.policy = "uniform"
                self.models[m] = pctx.partition(
                    num_chunks=1,
                    pipeline_size=pp_world_size,
                    rank=self.pp_rank,
                )
            else:
                self.models[m] = mc.model_fn(**mc.model_kwargs)
            self.models[m].eval()
            self.models[m].to("cpu", non_blocking=True)

        self.rpc_name = f"worker{self.global_rank}"
        rpc_options = {}
        if engine_config.rpc_disable_shm:
            # SHM may lead to timeout error. Disabling SHM and only enabling uv transport can solve this problem.
            # See https://discuss.pytorch.org/t/rpc-behavior-difference-between-pytorch-1-7-0-vs-1-9-0/124772/5
            # This is a workaround and may be solved in the future.
            rpc_options["_transports"] = ["uv"]
        trpc.init_rpc(
            self.rpc_name,
            rank=self.global_rank + 1,
            world_size=self.world_size + 1,
            rpc_backend_options=trpc.TensorPipeRpcBackendOptions(
                init_method=f"tcp://{engine_config.master_host}:{engine_config.rpc_port}",
                device_maps=build_device_maps(
                    self.world_size, n_proc_per_node, rank=self.global_rank
                ),
                **rpc_options,
            ),
        )
        self.to_master_pipe = Pipe(f"{self.global_rank}_to_m", self.rpc_name, "master")
        self.to_master_pipe.send(self.pp_rank)

        if self.pp_rank == 0:
            self.input_pipe = Pipe(
                f"m_to_{self.global_rank}", "master", self.rpc_name, max_size=engine_config.pipe_size
            )
        else:
            pp_prev_rank = gpc.get_prev_global_rank(ParallelMode.PIPELINE)
            self.input_pipe = Pipe(
                f"{pp_prev_rank}_to_{self.global_rank}",
                f"worker{pp_prev_rank}",
                self.rpc_name,
                max_size=engine_config.pipe_size,
            )
        if self.pp_rank == self.pp_world_size - 1:
            self.output_pipe = self.to_master_pipe
        else:
            pp_next_rank = gpc.get_next_global_rank(ParallelMode.PIPELINE)
            self.output_pipe = Pipe(
                f"{self.global_rank}_to_{pp_next_rank}",
                self.rpc_name,
                f"worker{pp_next_rank}",
                max_size=engine_config.pipe_size,
            )
        
        self.compute_stream = torch.cuda.default_stream()
        self.load_stream = torch.cuda.Stream()
        self.offload_stream = torch.cuda.Stream()
        self.wait_load_queue = Queue()
        self.wait_offload_queue = Queue()

        self.wait_load_thread = threading.Thread(target=self._wait_load_fn)
        self.wait_load_thread.daemon = True
        self.wait_offload_thread = threading.Thread(target=self._wait_offload_fn)
        self.wait_offload_thread.daemon = True
        self.wait_load_thread.start()
        self.wait_offload_thread.start()

        self.logger = get_dist_logger("computron")
        self.logger.info(f"{self.rpc_name} started")
        self._start()

    @contextmanager
    def _lifespan(self):
        try:
            yield
        finally:
            self._shutdown()

    def _start(self):
        with self._lifespan():
            while True:
                try:
                    entry = self.input_pipe.recv_nowait()
                    if isinstance(entry, TaskEntry):
                        with torch.inference_mode():
                            outputs = self._forward(entry.model_id, entry.batch)
                        self.output_pipe.send(TaskEntry(entry.uids, entry.model_id, outputs))
                    elif isinstance(entry, LoadEntry):
                        if self.pp_rank < (self.pp_world_size - 1):
                            self.output_pipe.send(entry)
                        with torch.inference_mode():
                            if entry.load:
                                with torch.cuda.stream(self.load_stream):
                                    self.models[entry.model_id].to("cuda", non_blocking=True)
                                event = self.load_stream.record_event()
                                self.wait_load_queue.put((entry, event))
                            else:
                                with torch.cuda.stream(self.offload_stream):
                                    self.models[entry.model_id].to("cpu", non_blocking=True)
                                event = self.offload_stream.record_event()
                                self.wait_offload_queue.put((entry, event))
                except RuntimeError:
                    time.sleep(0.01)

    def _shutdown(self) -> None:
        Terminator.shield()
        trpc.rpc_sync("master", Terminator.terminate)
        trpc.shutdown()

    def _forward(self, model_id: str, inputs: object) -> object:
        if isinstance(inputs, (tuple, list)):
            outputs = self.models[model_id](*inputs)
        elif isinstance(inputs, dict):
            outputs = self.models[model_id](**inputs)
        else:
            outputs = self.models[model_id](inputs)
        return outputs

    def _wait_load_fn(self):
        while True:
            if not self.wait_load_queue.empty():
                entry, load_event = self.wait_load_queue.get_nowait()
                load_event.synchronize()
                self.to_master_pipe.send(entry)
            else:
                time.sleep(0.02)

    def _wait_offload_fn(self):
        while True:
            if not self.wait_offload_queue.empty():
                entry, load_event = self.wait_offload_queue.get_nowait()
                load_event.synchronize()
                self.to_master_pipe.send(entry)
            else:
                time.sleep(0.02)
