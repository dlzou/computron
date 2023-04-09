from typing import List

import argparse
import asyncio
import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn

import mpu


class TwoSeries(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank = torch.tensor(rank, dtype=torch.float32)

    def forward(self, x):
        return x + torch.pow(2, self.rank)


class Controller:
    def __init__(self):
        pass

    def handle_request(self, x):
        dist.send(x, dst=0)


class ModelManager:
    def __init__(self, group):
        self.group = group
        group_rank = dist.get_rank(self.group)
        self.model = TwoSeries(group_rank)
        
    def handle_request(self):
        group_rank = dist.get_rank(self.group)
        x = torch.empty(1)
        if group_rank == 0:
            ctrl_rank = 2
            dist.recv(x, src=ctrl_rank)
            dist.barrier(group=self.group)
        else:
            dist.barrier(group=self.group)

        # Run pipeline
        if group_rank == 0:
            x = self.model(x)
            dist.send(x, dst=1, group=self.group)
        elif group_rank == 1:
            dist.recv(x, src=0, group=self.group)
            x = self.model(x)
            return x


def launch_controller():
    assert dist.is_initialized()
    ctrl = Controller()
    
    x = torch.tensor([1.])
    while True:
        x += 1
        ctrl.handle_request(x)
        time.sleep(3)


def launch_worker(model_group: dist.ProcessGroup):
    assert dist.is_initialized()
    mgr = ModelManager(model_group)
    
    while True:
        x = mgr.handle_request()
        print(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-models", "-m", type=int, required=True)
    parser.add_argument("--pp-size", "-p", type=int, required=True)
    parser.add_argument("--tp-size", "-t", type=int, required=True)
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    
    assert dist.is_available(), "pytorch.distributed not available"
    assert dist.is_gloo_available(), "GLOO not available"
    # assert dist.is_nccl_available(), "NCCL not available"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    mpu.init_mpu(args.num_models, args.pp_size, args.tp_size, "gloo")

    if rank == world_size - 1:
        launch_controller()
    else:
        launch_worker()
        