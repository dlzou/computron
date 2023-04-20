import asyncio
import time

from launch import launch_multi_model
from models import echo


ctlr = None


async def make_request(num_reqs):
    pass


if __name__ == "__main__":
    ctlr = launch_multi_model(
        model_ids=["echo0", "echo1"],
        request_types=[echo.MLPRequest, echo.MLPRequest],
        unpack_request_fns=[echo.unpack_request, echo.unpack_request],
        pack_response_fns=[echo.pack_response, echo.pack_response],
        model_fns=[echo.MLP, echo.MLP]
    )
