import asyncio
from asyncio import Lock
from collections import deque
import pickle
import signal
import sys
from threading import Thread
import time
from typing import Optional

from pydantic import BaseModel, Field


async def send_obj(writer, obj):
    bytes = pickle.dumps(obj)
    writer.write(b'%d\n' % len(bytes))
    writer.write(bytes)
    await writer.drain()


async def recv_obj(reader):
    prefix = await reader.readline()
    msglen = int(prefix)
    bytes = await reader.readexactly(msglen)
    return pickle.loads(bytes)


class GenerationTaskReq(BaseModel):
    max_tokens: int = Field(default=64, gt=0, le=256, example=64)
    prompt: str = Field(default='Question: Where were the 2004 Olympics held?\nAnswer: Athens, Greece\n\nQuestion: What is the longest river on the earth?\nAnswer:',
        min_length=1, example='Question: Where were the 2004 Olympics held?\nAnswer: Athens, Greece\n\nQuestion: What is the longest river on the earth?\nAnswer:')
    top_k: Optional[int] = Field(default=50, gt=0, example=50)
    top_p: Optional[float] = Field(default=0.5, gt=0.0, lt=1.0, example=0.5)
    temperature: Optional[float] = Field(default=0.7, gt=0.0, lt=1.0, example=0.7)


def print_req(req: GenerationTaskReq):
    print(req.max_tokens)
    print(req.prompt)
    print(req.top_k)
    print(req.top_p)
    print(req.temperature)


class Engine:
    def __init__(self, request_port):
        self.request_port = request_port
        self.submit_thread = None
        self.submit_queue = deque()
        self.completion_queue = deque()
        self._start()

    def _start(self):
        loop = asyncio.new_event_loop()
        shutdown_signals = (signal.SIGINT, signal.SIGTERM)
        for s in shutdown_signals:
            loop.add_signal_handler(s, lambda: loop.create_task(self._shutdown(loop)))
        try:
            loop.create_task(self._request_server())
            loop.create_task(self._submit_loop())
            loop.run_forever()
        finally:
            loop.close()

    async def _shutdown(self, loop):
        tasks = [t for t in asyncio.all_tasks() if t is not
             asyncio.current_task()]
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    async def _handle_request(self, reader, writer):
        obj = await recv_obj(reader)
        self.submit_queue.append(obj)
        while len(self.completion_queue) == 0:
            await asyncio.sleep(0)
        obj = self.completion_queue.popleft()
        await send_obj(writer, obj)
        writer.close()
        await writer.wait_closed()

    async def _request_server(self):
        server = await asyncio.start_server(
            lambda r, w: self._handle_request(r, w),
            "127.0.0.1", self.request_port
        )
        async with server:
            await server.serve_forever()

    async def _submit_loop(self):
        while True:
            if len(self.submit_queue) > 0:
                req = self.submit_queue.popleft()
                print_req(req)
                self.completion_queue.append(req)
            else:
                await asyncio.sleep(0)
    
    # async def _completion_loop(self):
    #     while True:
    #         if len(self.completion_queue) == 0:
    #             await asyncio.sleep(0.01)


async def client(port):
    start_time = time.time()
    reader, writer = await asyncio.open_connection(
        '127.0.0.1', port)
    
    print(f"Connection took {time.time() - start_time}s")

    req = GenerationTaskReq()
    await send_obj(writer, req)
    
    obj = await recv_obj(reader)
    print_req(obj)
    
    print("Closing connection")
    writer.close()
    await writer.wait_closed()

    print(f"Round trip took {time.time() - start_time}s")


if __name__ == "__main__":
    port = 29600
    if sys.argv[1] == "client":
        asyncio.run(client(port))
    elif sys.argv[1] == "engine":
        engine = Engine(port)
