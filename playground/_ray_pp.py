from dataclasses import dataclass
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List

import os
import ray
import ray.util.collective as rc
import torch
import torch.nn as nn

import mlp


class TwoSeries(nn.Module):
    def __init__(self, rank):
        self.rank = rank

    def forward(self, x):
        return x + 2 ** self.rank


@dataclass
class PPGroup:
    group_name: str
    world_size: int
    rank: int
    prev_rank: int
    next_rank: int


@ray.remote(num_gpus=1)
class StageWorker:
    def __init__(self, stage: nn.Module, pg: PPGroup):
        self.stage = stage
        self.pg = pg
        self.input = None

    def set_input(self, input_):
        self.input = input_
        
    def recv_forward(self):
        # Dunno if data_ptr is correct here
        assert self.pg.prev_rank != -1, "no stage to recv from"
        rc.recv(self.input.data_ptr(), self.pg.prev_rank, self.pg.group_name)
    
    def run_stage_forward(self):
        output = self.stage(self.input)
        if self.pg.next_rank != -1:
            rc.send(output.data_ptr(), self.pg.next_rank, self.pg.group_name)

    def get_device(self):
        print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))


test_set_size = 2 ** 16
batch_size = 256
ubatch_size = 16


def run_pipeline(stages: List[StageWorker], batch):
    ubatches = iter(batch.split(ubatch_size, dim=0))

    num_warmup_rounds = len(stages) - 1
    for i in range(num_warmup_rounds):
        x = next(ubatches)
        stages[0].set_input(x)


if __name__ == "__main__":
    ray.init(num_gpus=2)
    num_stages = 2

    m1 = TwoSeries(rank=0)
    m2 = TwoSeries(rank=1)

    pp1 = PPGroup(
        group_name="TwoSeriesPP",
        world_size=2,
        rank=0,
        pp_prev_rank=-1,
        pp_next_rank=1,
    )
    w1 = StageWorker.remote(m1, pp1)
    pp2 = PPGroup(
        group_name="TwoSeriesPP",
        world_size=2,
        rank=1,
        pp_prev_rank=0,
        pp_next_rank=-1,
    )
    w2 = StageWorker.remote(m2, pp2)
    print(ray.get(w1.get_device.remote()))
    print(ray.get(w2.get_device.remote()))
    
    stages = [w1, w2]
    options = {
        "group_name": "TwoSeriesPP",
        "world_size": len(stages),
        "ranks": range(len(stages)),
        "backend": "nccl",
    }
    rc.declare_collective_group(stages, **options)

    inputs = torch.randn(test_set_size, 1)
    test_set = TensorDataset(inputs)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    