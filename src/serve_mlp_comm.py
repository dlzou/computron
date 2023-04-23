import asyncio
from functools import partial
import time

from energonai import launch_engine
import torch

# Should install as package instead
from models import mlp


engine = None


async def make_requests(num_reqs):
    start_time = time.time()
    for i in range(num_reqs):
        data = torch.ones((256,)) * i
        uid = id(data)
        engine.submit(uid, data)
        output = await engine.wait(uid)
        print(f"Response time {i}: {time.time() - start_time}")
        # print(output)
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    tp_world_size = 2
    pp_world_size = 1
    
    engine = launch_engine(
        tp_world_size=tp_world_size,
        pp_world_size=pp_world_size,
        master_host="localhost",
        master_port=29600,
        rpc_port=29601,
        model_fn=partial(mlp.MLP, dim=256),
    )

    time.sleep(15) # Wait for engine to start
    asyncio.run(make_requests(20))
