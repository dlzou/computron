from collections import deque
from typing import Any, Hashable, Iterable

import colossalai.nn as col_nn
from computron import BatchManager, SubmitEntry, TaskEntry
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, offset, dim: int = 256):
        super().__init__()
        intermediate_dim = dim * 4
        self.dense_1 = col_nn.Linear1D_Col(dim, intermediate_dim)
        self.activation = torch.nn.GELU()
        self.dense_2 = col_nn.Linear1D_Row(intermediate_dim, dim)
        self.dropout = col_nn.Dropout(0.1)
        self.offset = offset

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class MLPBatchManager(BatchManager):
    def __init__(self, max_batch_size: int = 1):
        self.max_batch_size = max_batch_size

    def make_batch(
        self, q: deque[SubmitEntry],
    ) -> tuple[TaskEntry, dict]:
        entry = q.popleft()
        uids = [entry.uid]
        model_id = entry.model_id
        batch = [entry.data]
        while len(batch) < self.max_batch_size:
            if len(q) == 0:
                break
            entry = q.popleft()
            uids.append(entry.uid)
            batch.append(entry.data)
        inputs = torch.stack(batch)
        return TaskEntry(tuple(uids), model_id, inputs), {}

    def split_batch(self, task_entry: TaskEntry) -> Iterable[tuple[Hashable, Any]]:
        ret = []
        for uid, output in zip(task_entry.uids, task_entry.batch):
            ret.append((uid, output))
        return ret
