from dataclasses import dataclass
from typing import Hashable

from pydantic import BaseModel


class OffloadRequest(BaseModel):
    loaded: bool


class OffloadResponse(BaseModel):
    success: bool


@dataclass
class OffloadEntry:
    uid: Hashable
    loaded: bool
