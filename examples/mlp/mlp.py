from dataclasses import dataclass
from typing import Any, Deque, Hashable, List, Tuple, Union

import colossalai.nn as col_nn
from colossalai.utils import print_rank_0
from computron import OffloadEntry, OffloadingBatchManager
from energonai import SubmitEntry, TaskEntry
from pydantic import BaseModel
import torch
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


exec_seq = ["dense_1", "activation", "dense_2", "dropout"]


class MLPRequest(BaseModel):
    data: Any


def unpack_request(req: MLPRequest) -> SubmitEntry:
    return SubmitEntry(id(req), req.data)


class MLPResponse(BaseModel):
    output: Any


def pack_response(output: Any) -> MLPResponse:
    return MLPResponse(output=output)


# TODO: write own parent OffloadingBatchManager
class MLPBatchManager(OffloadingBatchManager):
    def __init__(self, max_batch_size: int = 1):
        self.max_batch_size = max_batch_size

    def make_batch(
        self, q: Deque[Union[SubmitEntry, OffloadEntry]]
    ) -> Tuple[Union[TaskEntry, OffloadEntry], dict]:
        entry = q.popleft()
        if isinstance(entry, OffloadEntry):
            return entry, {}

        uids = [entry.uid]
        batch = [entry.data]
        while len(batch) < self.max_batch_size:
            if len(q) == 0:
                break
            if isinstance(q[0], OffloadEntry):
                break
            entry = q.popleft()
            uids.append(entry.uid)
            batch.append(entry.data)
        inputs = torch.stack(batch)
        return TaskEntry(tuple(uids), inputs), {}
    
    def split_batch(self, task_entry: TaskEntry) -> List[Tuple[Hashable, Any]]:
        ret = []
        for uid, output in zip(task_entry.uids, task_entry.batch):
            ret.append((uid, output))
        return ret
