from dataclasses import dataclass
import time
from typing import Any

from colossalai.utils import print_rank_0
from energonai import SubmitEntry
from pydantic import BaseModel
import torch
import torch.nn as nn


class Echo(nn.Module):
    def __init__(self):
        super().__init__()
        print_rank_0("Intializing echo model")
        self.param = torch.zeros((1,))

    def forward(self, x):
        print_rank_0("Executing echo model")
        time.sleep(1)
        return x


class EchoRequest(BaseModel):
    data: Any


def unpack_request(req: EchoRequest) -> SubmitEntry:
    return SubmitEntry(id(req), req.data)


class EchoResponse(BaseModel):
    output: Any


def pack_response(output: Any) -> EchoResponse:
    return EchoResponse(output=output)
