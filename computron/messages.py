from dataclasses import dataclass
from typing import Hashable

from pydantic import BaseModel


class PingRequest(BaseModel):
    pass


class PingResponse(BaseModel):
    pass


class OffloadRequest(BaseModel):
    load: bool
    flush: bool


class OffloadResponse(BaseModel):
    success: bool


@dataclass
class OffloadEntry:
    uid: Hashable
    load: bool
    flush: bool
