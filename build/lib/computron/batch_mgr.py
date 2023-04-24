from typing import Deque, Tuple, Union

from energonai import BatchManager, SubmitEntry, TaskEntry

from computron.offload import OffloadEntry


class OffloadingBatchManager(BatchManager):
    def make_batch(
        self, q: Deque[Union[SubmitEntry, OffloadEntry]]
    ) -> Tuple[Union[TaskEntry, OffloadEntry], dict]:
        entry = q.popleft()
        if isinstance(entry, OffloadEntry):
            return entry, {}
        return TaskEntry((entry.uid,), entry.data), {}
