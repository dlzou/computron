from typing import Any, Callable, Dict, Optional

from energonai import BatchManager, launch_engine
import torch.multiprocessing as mp
import torch.nn as nn

from batch import BatchManagerForGeneration
from engine import OffloadingEngine
from worker import OffloadingWorker
from models.mlp import MLP


def launch_swap_workers():
    pass


def launch_swap_engine(
    tp_world_size: int,
    pp_world_size: int,
    master_host: str,
    master_port: int,
    rpc_port: int,
    model_fn: Callable[[Any], nn.Module],
    n_nodes: int = 1,
    node_rank: int = 0,
    batch_manager: Optional[BatchManager] = None,
    pipe_size: int = 1,
    queue_size: int = 0,
    rpc_disable_shm:
    bool = True,
    **model_kwargs: Any,
) -> Optional[OffloadingEngine]:
    world_size = tp_world_size * pp_world_size
    assert world_size % n_nodes == 0
    n_proc_per_node = world_size // n_nodes
    launch_swap_workers(
        tp_world_size,
        pp_world_size,
        master_host,
        master_port,
        rpc_port,
        model_fn,
        n_proc_per_node,
        node_rank,
        pipe_size,
        **model_kwargs,
    )
    if node_rank == 0:
        engine = OffloadingEngine(
            tp_world_size,
            pp_world_size,
            master_host,
            rpc_port,
            n_proc_per_node,
            batch_manager,
            pipe_size,
            queue_size,
            rpc_disable_shm,
        )
        return engine


def get_model_fn():
    return MLP


def launch_multi_model():
    num_models = 2
    tp_world_size = 2
    pp_world_size = 1
    master_host = "localhost" # "127.0.0.1"
    first_port = 29600
    master_ports = [(first_port + 2*m) for m in range(num_models)]
    rpc_ports = [(first_port + 2*m + 1) for m in range(num_models)]
    model_fn = get_model_fn()
    n_nodes = 1
    node_rank = 0
    batch_manager = BatchManager()
    pipe_size = 1
    queue_size = 1
    rpc_disable_shm = True
    model_kwargs = {}

    mp.set_start_method("spawn")
    processes = []
    for m in range(num_models):
        p = mp.Process(
            target=launch_engine,
            args=(
                tp_world_size,
                pp_world_size,
                master_host,
                master_ports[m],
                rpc_ports[m],
                model_fn,
                n_nodes,
                node_rank,
                batch_manager,
                pipe_size,
                queue_size,
                rpc_disable_shm,
            ),
            kwargs=model_kwargs,
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    launch_multi_model()
