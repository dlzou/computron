from collections import deque
from collections.abc import Hashable, Iterable

from computron.messages import SubmitEntry, TaskEntry


class BatchManager:
    def make_batch(self, q: deque[SubmitEntry]) -> tuple[TaskEntry, dict]:
        """Join multiple task entries to be submitted to workers as a batch."""
        entry = q.popleft()
        return TaskEntry((entry.uid,), entry.model_id, entry.data), {}

    def split_batch(self, task_entry: TaskEntry, **kwargs) -> Iterable[tuple[Hashable, object]]:
        """Split output data in a batch by original uid."""
        return [(task_entry.uids[0], task_entry.batch)]
