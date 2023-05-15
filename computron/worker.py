import time
from contextlib import contextmanager
from typing import Any, Callable, List

import colossalai
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
# from colossalai.pipeline.pipelinable import PipelinableContext
from energonai.pipe import Pipe
from energonai.task import TaskEntry
from energonai.utils import build_device_maps, Terminator
import torch
import torch.distributed.rpc as trpc
import torch.nn as nn

from computron.launch import EngineConfig, ModelConfig
from computron.messages import LoadEntry


class Worker:
    def __init__(
        self,
        rank: int,
        tp_world_size: int,
        pp_world_size: int,
        n_proc_per_node: int,
        engine_config: EngineConfig,
        model_configs: List[ModelConfig],
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

        # if pipelinable:
        #     pctx = PipelinableContext()
        #     with pctx:
        #         self.model = model_fn(**model_kwargs)
        #     pctx.to_layer_list()
        #     pctx.policy = "uniform"
        #     self.model: nn.Module = pctx.partition(
        #         num_chunks=1,
        #         pipeline_size=pp_world_size,
        #         rank=self.pp_rank,
        #     )
        # else:
        #     self.model: nn.Module = model_fn(**model_kwargs)
        # self.model.eval()
        # self.model.to("cpu", non_blocking=True)
        # # self.model._apply(lambda t: t.pin_memory())
        # torch.cuda.empty_cache()

        self.models = {}
        for mc in model_configs:
            self.models[mc.model_id] = mc.model_fn(**mc.model_kwargs)
            self.models[mc.model_id].eval()
            self.models[mc.model_id].cpu().pin_memory()

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
        
        self.model_stream = torch.cuda.default_stream()
        self.load_stream = torch.cuda.Stream()

        self.logger = get_dist_logger("computron")
        self.logger.info(f"{self.rpc_name} start")
        self.logger.info(f"{self.rpc_name} dtype: {next(self.model.parameters()).dtype}")
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
                            self.logger.info(f"worker loaded={next(self.model.parameters()).is_cuda}")
                            self.logger.info(f"worker allocated={torch.cuda.memory_allocated() / (10 ** 9)}")
                            outputs = self._forward(entry.batch)
                        self.output_pipe.send(TaskEntry(entry.uids, outputs))
                    elif isinstance(entry, LoadEntry):
                        self.output_pipe.send(entry)
                        with torch.inference_mode():
                            if entry.load:
                                self.model.to("cuda", non_blocking=True)
                                # self.model._apply(lambda t: t.cuda())
                            else:
                                self.model.to("cpu", non_blocking=True)
                                # self.model._apply(lambda t: t.pin_memory())
                                torch.cuda.empty_cache()
                        self.to_master_pipe.send(entry)
                except RuntimeError:
                    time.sleep(0.01)

    def _shutdown(self) -> None:
        Terminator.shield()
        trpc.rpc_sync("master", Terminator.terminate)
        trpc.shutdown()

    def _forward(self, inputs: Any) -> Any:
        if isinstance(inputs, (tuple, list)):
            outputs = self.model(*inputs)
        elif isinstance(inputs, dict):
            outputs = self.model(**inputs)
        else:
            outputs = self.model(inputs)
        return outputs

    def _load_thread(self):
        pass