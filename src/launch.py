from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from energonai import BatchManager, SubmitEntry
from pydantic import BaseModel
import torch.multiprocessing as mp

from controller import Controller
from engine import OffloadingEngine
from worker import OffloadingWorker
from models import mlp


@dataclass
class ModelConfig:
    model_id: str
    master_host: str
    master_port: int
    rpc_port: int
    request_port: int
    request_type: BaseModel
    unpack_request_fn: Callable[[BaseModel], SubmitEntry]
    pack_response_fn: Callable[[Any], BaseModel]
    model_fn: Callable
    # Can only partition for PP once colossalai is launched and gpc is set.
    # Lazy solution here is to pass the execution sequence to workers and let them partition.
    # If model_exec_seq = None, use model_fn as given.
    model_exec_seq: Optional[List[Any]] = None
    batch_manager: Optional[BatchManager] = None
    pipe_size: int = 1
    queue_size: int = 0
    rpc_disable_shm: bool = True
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


def launch_offloading_workers(
    config: ModelConfig,
    tp_world_size: int,
    pp_world_size: int,
    n_proc_per_node: int = 1,
    node_rank: int = 0, 
):
    ctx = mp.get_context("spawn")
    procs = []
    for i in range(n_proc_per_node):
        rank = n_proc_per_node * node_rank + i
        p = ctx.Process(
            target=OffloadingWorker,
            args=(
                rank,
                tp_world_size,
                pp_world_size,
                config.master_host,
                config.master_port,
                config.rpc_port,
                n_proc_per_node,
                config.model_fn,
                config.model_exec_seq,
                config.pipe_size,
                config.rpc_disable_shm,
            ),
            kwargs=config.model_kwargs,
        )
        procs.append(p)
        p.start()


def launch_multi_model(
    model_configs: List[ModelConfig],
    tp_world_size: int,
    pp_world_size: int,
    n_nodes: int,
    node_rank: int,
) -> Optional[OffloadingEngine]:
    num_models = len(model_configs)
    world_size = tp_world_size * pp_world_size
    assert world_size % n_nodes == 0
    n_proc_per_node = world_size // n_nodes

    ctx = mp.get_context("spawn")
    procs = []
    for i in range(num_models):
        config = model_configs[i]
        launch_offloading_workers(config, tp_world_size, pp_world_size, n_proc_per_node, node_rank)
        if node_rank == 0:
            p = ctx.Process(
                target=OffloadingEngine,
                args=(
                    tp_world_size,
                    pp_world_size,
                    config.master_host,
                    config.rpc_port,
                    config.request_port,
                    config.request_type,
                    config.unpack_request_fn,
                    config.pack_response_fn,
                    n_proc_per_node,
                    config.batch_manager,
                    config.pipe_size,
                    config.queue_size,
                    config.rpc_disable_shm,
                )
            )
            procs.append(p)
            p.start()

    if node_rank == 0:
        ctlr = Controller()
        for i in range(num_models):
            config = model_configs[i]
            ctlr.register_model(config.model_id, config.master_host, config.request_port)
        return ctlr

    # TODO: add signal handler that syncs with engines and workers


if __name__ == "__main__":
    # Basic launch and shutdown test
    num_models = 2
    first_port = 29600
    configs = []
    for i in range(num_models):
        config = ModelConfig(
            model_id=f"mlp{i}",
            master_host="localhost",
            master_port=(first_port + 3*i),
            rpc_port=(first_port + 3*i + 1),
            request_port=(first_port + 3*i + 2),
            request_type=mlp.MLPRequest,
            unpack_request_fn=mlp.unpack_request,
            pack_response_fn=mlp.pack_response,
            model_fn=mlp.MLP,
        )
        configs.append(config)

    launch_multi_model(
        configs,
        tp_world_size=2,
        pp_world_size=1,
        n_nodes=1,
        node_rank=0,
    )
