from collections.abc import Hashable
from dataclasses import dataclass

from pydantic import BaseModel


class PingRequest(BaseModel):
    pass


class PingResponse(BaseModel):
    pass


@dataclass
class SubmitEntry:
    uid: Hashable
    model_id: str
    data: object


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
