from collections.abc import Hashable
from dataclasses import dataclass


@dataclass
class SubmitEntry:
    uid: Hashable
    model_id: str
    data: object
    ts: object


@dataclass
class TaskEntry:
    uids: tuple[Hashable, ...]
    model_id: str
    batch: object


@dataclass
class LoadEntry:
    uid: Hashable
    model_id: str
    load: bool
