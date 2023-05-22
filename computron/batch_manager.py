from collections import deque
from typing import Any, Hashable, Iterable

# from energonai import BatchManager, SubmitEntry, TaskEntry

from computron.messages import SubmitEntry, TaskEntry


# class OffloadingBatchManager(BatchManager):
#     def make_batch(
#         self, q: Deque[Union[SubmitEntry, LoadEntry]]
#     ) -> Tuple[Union[TaskEntry, LoadEntry], dict]:
#         entry = q.popleft()
#         if isinstance(entry, LoadEntry):
#             return entry, {}
#         return TaskEntry((entry.uid,), entry.data), {}


class BatchManager:
    def make_batch(self, q: deque[SubmitEntry]) -> tuple[TaskEntry, dict]:
        """Join multiple task entries to be submitted to workers as a batch."""
        entry = q.popleft()
        return TaskEntry((entry.uid,), entry.model_id, entry.data), {}

    def split_batch(self, task_entry: TaskEntry, **kwargs) -> Iterable[tuple[Hashable, Any]]:
        """Split output data in a batch by original uid."""
        return [(task_entry.uids[0], task_entry.batch)]
