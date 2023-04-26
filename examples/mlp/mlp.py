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
        self.dense_1 = col_nn.Linear1D_Col(dim, intermediate_dim)
        self.activation = torch.nn.GELU()
        self.dense_2 = col_nn.Linear1D_Row(intermediate_dim, dim)
        self.dropout = col_nn.Dropout(0.1)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class MLPRequest(BaseModel):
    data: Any


class MLPResponse(BaseModel):
    output: Any


def unpack_request(req: MLPRequest) -> SubmitEntry:
    return SubmitEntry(id(req), req.data)


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
