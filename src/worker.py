import time
from typing import Any, Callable, List

from colossalai.pipeline.pipelinable import PipelinableContext
from energonai.task import TaskEntry
from energonai.worker import Worker
import torch
import torch.nn as nn


class OffloadingWorker(Worker):
    def __init__(
        self,
        rank: int,
        tp_world_size: int,
        pp_world_size: int,
        master_host: str,
        master_port: int,
        rpc_port: int,
        n_proc_per_node: int,
        model_fn: Callable[[Any], nn.Module],
        model_exec_seq: List[Any],
        pipe_size: int = 1,
        rpc_disable_shm: bool = True,
        **model_kwargs: Any
    ):
        super().__init__(
            rank, tp_world_size, pp_world_size, master_host, master_port, rpc_port, n_proc_per_node,
            model_fn, pipe_size, rpc_disable_shm, **model_kwargs
        )
        if model_exec_seq is not None:
            pctx = PipelinableContext()
            with pctx:
                model_fn(**model_kwargs)
            pctx.to_layer_list(model_exec_seq)
            self.model: nn.Module = pctx.partition(
                num_chunks=1,
                pipeline_size=pp_world_size,
                rank=self.pp_rank,
            ).cuda()

    def _start(self):
        with self._lifespan():
            while True:
                try:
                    entry = self.input_pipe.recv_nowait()
                    if isinstance(entry, TaskEntry):
                        with torch.inference_mode():
                            outputs = self._forward(entry.batch)
                        self.output_pipe.send(TaskEntry(entry.uids, outputs))
                    # elif hasattr(entry, OffloadEntry):
                    #     if entry.on_device:
                    #         self.model.to("cuda")
                    #     else:
                    #         self.model.to("cpu")
                except RuntimeError:
                    time.sleep(0.01)