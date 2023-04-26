import os
import sys

import colossalai
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
import colossalai.nn as col_nn
from colossalai.pipeline.pipelinable import PipelinableContext
from colossalai.utils import get_current_device, print_rank_0
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        intermediate_dim = dim * 4
        self.dense_1 = col_nn.Linear(dim, intermediate_dim)
        self.activation = torch.nn.GELU()
        self.dense_2 = col_nn.Linear(intermediate_dim, dim)
        self.dropout = col_nn.Dropout(0.1)
        print_rank_0("Initialized MLP")

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


TP_WORLD_SIZE = 2
PP_WORLD_SIZE = 2

config = dict(parallel=dict(
    data=1,
    pipeline=PP_WORLD_SIZE,
    tensor=dict(size=TP_WORLD_SIZE, mode='1d'),
))

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

colossalai.launch(
    config=config,
    rank=rank,
    local_rank=local_rank,
    world_size=world_size,
    host="localhost",
    port=29600,
    backend="nccl",
)

pctx = PipelinableContext()
with pctx:
    m = MLP()
# pctx.to_layer_list()
pctx.policy = "uniform"
m = pctx.partition(
    num_chunks=1,
    pipeline_size=PP_WORLD_SIZE,
    rank=gpc.get_local_rank(ParallelMode.PIPELINE),
)

# if gpc.is_pipeline_first_stage():


# x = torch.randn((16, 256), device=get_current_device())
# torch.distributed.broadcast(x, src=0)  # synchronize input

print(f"{rank}\n", m)