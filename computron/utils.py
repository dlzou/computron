import asyncio
import pickle


async def send_obj(writer: asyncio.StreamWriter, obj: object):
    bytes = pickle.dumps(obj)
    writer.write(b"%d\n" % len(bytes))
    writer.write(bytes)
    await writer.drain()


async def recv_obj(reader: asyncio.StreamReader):
    header = await reader.readline()
    msg_len = int(header)
    bytes = await reader.readexactly(msg_len)
    return pickle.loads(bytes)
