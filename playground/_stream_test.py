import asyncio
import pickle
import sys
import time
from typing import Optional

from pydantic import BaseModel, Field


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


async def send_bytes(bytes, writer):
    writer.write(b'%d\n' % len(bytes))
    writer.write(bytes)
    await writer.drain()


async def recv_bytes(reader):
    prefix = await reader.readline()
    msglen = int(prefix)
    return await reader.readexactly(msglen)


async def send_obj(obj, writer):
    bytes = pickle.dumps(obj)
    writer.write(b'%d\n' % len(bytes))
    writer.write(bytes)
    await writer.drain()


async def recv_obj(reader):
    prefix = await reader.readline()
    msglen = int(prefix)
    bytes = await reader.readexactly(msglen)
    return pickle.loads(bytes)


async def client(port):
    start_time = time.time()
    reader, writer = await asyncio.open_connection(
        '127.0.0.1', port)
    
    print(f"Connection took {time.time() - start_time}s")

    # msg_bytes = str.encode("hello world")
    req = GenerationTaskReq()
    await send_obj(req, writer)
    
    # bytes = await recv_bytes(reader)
    # print("Received:", bytes.decode())
    obj = await recv_obj(reader)
    print_req(obj)
    
    print("Closing connection")
    writer.close()
    await writer.wait_closed()

    print(f"Round trip took {time.time() - start_time}s")


async def handle_request(reader, writer):
    # bytes = await recv_bytes(reader)
    # print("Received:", bytes.decode())
    obj = await recv_obj(reader)
    print_req(obj)
    
    # msg_bytes = str.encode("goodbye world")
    req = GenerationTaskReq()
    await send_obj(req, writer)

    print("Closing connection")
    writer.close()
    await writer.wait_closed()


async def server(port):
    server = await asyncio.start_server(
        handle_request, '127.0.0.1', port)

    addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
    print(f'Serving on {addrs}')

    async with server:
        await server.serve_forever()
        

# async def server(port):
#     reader, writer = await asyncio.open_connection(
#         '127.0.0.1', port)
    
#     while True:
#         obj = await recv_obj(reader)
#         print_req(obj)

#         req = GenerationTaskReq()
#         await send_obj(req, writer)
    
    # print("Closing connection")
    # writer.close()
    # await writer.wait_closed()


if __name__ == "__main__":
    port = 29600
    if sys.argv[1] == "client":
        asyncio.run(client(port))
    elif sys.argv[1] == "server":
        asyncio.run(server(port))
