import time

from colossalai.utils import print_rank_0
import torch.nn as nn


class Echo(nn.Module):
    def __init__(self):
        super().__init__()
        print_rank_0("Initializing Echo")

    def forward(self, x):
        print_rank_0("Executing Echo")
        time.sleep(1)
        return x
