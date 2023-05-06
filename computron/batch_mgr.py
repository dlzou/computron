from typing import Deque, Tuple, Union

from energonai import BatchManager, SubmitEntry, TaskEntry

from computron.messages import LoadEntry


class OffloadingBatchManager(BatchManager):
    def make_batch(
        self, q: Deque[Union[SubmitEntry, LoadEntry]]
    ) -> Tuple[Union[TaskEntry, LoadEntry], dict]:
        entry = q.popleft()
        if isinstance(entry, LoadEntry):
            return entry, {}
        return TaskEntry((entry.uid,), entry.data), {}
