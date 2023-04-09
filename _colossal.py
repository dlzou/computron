from colossalai.utils import get_current_device, print_rank_0
from typing import Callable, Dict

import colossalai
import colossalai.nn as col_nn
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        intermediate_dim = dim * 4
        self.dense_1 = col_nn.Linear(dim, intermediate_dim)
        print_rank_0(f'Weight of the first linear layer: {self.dense_1.weight.transpose(0, 1).shape}')
        self.activation = torch.nn.GELU()
        self.dense_2 = col_nn.Linear(intermediate_dim, dim)
        print_rank_0(f'Weight of the second linear layer: {self.dense_2.weight.transpose(0, 1).shape}')
        self.dropout = col_nn.Dropout(0.1)

    def forward(self, x):
        x = self.dense_1(x)
        print_rank_0(f'Output of the first linear layer: {x.shape}')
        x = self.activation(x)
        x = self.dense_2(x)
        print_rank_0(f'Output of the second linear layer: {x.shape}')
        x = self.dropout(x)
        return x


def run_colossal():
    m = MLP()
    
    x = torch.randn((16, 256), device=get_current_device())
    torch.distributed.broadcast(x, src=0)  # synchronize input
    x = m(x)


class PlusConst(nn.Module):
    def __init__(self, const):
        super().__init__()
        self.const = const

    def forward(self, x):
        return x + self.const


def run(config):
    assert dist.is_initialized(), "pytorch.distributed not initialized"
    rank = dist.get_rank()

    x = torch.ones(2, 2)
    if rank == 0:
        model = PlusConst(1)
        x = model(x)
        dist.send(x, dst=1)
    elif rank == 1:
        model = PlusConst(2)
        dist.recv(x, src=0)
        x = model(x)
        print(x)


def init_process(
    port: int,
    rank: int,
    world_size: int,
    config: Dict,
    fn: Callable,
    backend: str = "nccl",
):
    """ Initialize the distributed environment. """
    # os.environ['MASTER_ADDR'] = "127.0.0.1"
    # os.environ['MASTER_PORT'] = str(port)
    # dist.init_process_group(backend, rank, world_size)
    # fn(config)
    
    colossalai.launch(
        config=config,
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        host="127.0.0.1",
        port=port,
        backend=backend,
    )
    fn()


if __name__ == "__main__":
    num_models = 1
    pp_size = 1
    tp_size = 2
    model_world_size = pp_size * tp_size

    config = dict(parallel=dict(
        pipeline=pp_size,
        tensor=dict(size=tp_size, mode='1d'),
        data=1,
    ))

    mp.set_start_method("spawn")
    processes = []
    first_port = 29500
    model_ports = [first_port + m for m in range(num_models)]
    for m in range(num_models):
        for rank in range(model_world_size):
            p = mp.Process(
                target=init_process,
                args=(model_ports[m], rank, model_world_size, config, run_colossal, "gloo")
            )
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

# if __name__ == "__main__":
#     config = dict(parallel=dict(
#         pipeline=1,
#         tensor=dict(size=2, mode='1d'),
#         data=1,
#     ))

#     world_size = int(os.environ["WORLD_SIZE"])
#     rank = int(os.environ["RANK"])
#     init_process(29500, rank, world_size, config, run_colossal, "gloo")
