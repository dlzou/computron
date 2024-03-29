import os

import torch.multiprocessing as mp

from computron.config import EngineConfig, ModelConfig
from computron.engine import Engine
from computron.worker import Worker


def _launch_workers(
    engine_config: EngineConfig,
    model_configs: list[ModelConfig],
    tp_world_size: int,
    pp_world_size: int,
    n_proc_per_node: int,
    node_rank: int,
    log_dir: str | None = None,
):
    ctx = mp.get_context("spawn")
    procs = []
    for i in range(n_proc_per_node):
        rank = n_proc_per_node * node_rank + i
        p = ctx.Process(
            target=Worker,
            args=(
                engine_config,
                model_configs,
                rank,
                tp_world_size,
                pp_world_size,
                n_proc_per_node,
                log_dir,
            ),
        )
        procs.append(p)
        p.start()


def launch_computron(
    engine_config: EngineConfig,
    model_configs: list[ModelConfig],
    tp_world_size: int,
    pp_world_size: int,
    n_nodes: int = 1,
    node_rank: int = 0,
    log_dir: str | None = None,
) -> Engine | None:
    world_size = tp_world_size * pp_world_size
    assert world_size % n_nodes == 0
    n_proc_per_node = world_size // n_nodes

    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    _launch_workers(
        engine_config,
        model_configs,
        tp_world_size,
        pp_world_size,
        n_proc_per_node,
        node_rank,
        log_dir,
    )
    if node_rank == 0:
        return Engine(
            engine_config,
            model_configs,
            tp_world_size,
            pp_world_size,
            n_proc_per_node,
            log_dir,
        )
