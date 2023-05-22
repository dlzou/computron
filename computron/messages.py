from dataclasses import dataclass
from typing import Any, Hashable, Tuple

from pydantic import BaseModel


class PingRequest(BaseModel):
    pass


class PingResponse(BaseModel):
    pass


# class LoadRequest(BaseModel):
#     model_id: str
#     load: bool
#     blocking: bool


# class LoadResponse(BaseModel):
#     success: bool


@dataclass
class SubmitEntry:
    uid: Hashable
    model_id: str
    data: Any


@dataclass
class TaskEntry:
    uids: Tuple[Hashable, ...]
    model_id: str
    batch: Any


@dataclass
class LoadEntry:
    uid: Hashable
    model_id: str
    load: bool
    blocking: bool
