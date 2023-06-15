import asyncio
import os
import time
import torch

from computron.models import vit
from energonai import launch_engine

from proc_img import proc_img


async def make_requests(num_reqs):
    start_time = time.time()
    for i in range(num_reqs):
        img = proc_img(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "dataset/n01667114_9985.JPEG",
            )
        )
        data = torch.unsqueeze(img, 0)
        uid = id(data)
        engine.submit(uid, data)
        output = await engine.wait(uid)
        print(f"Response time {i}: {time.time() - start_time}")
        print(output.shape)
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    tp_world_size = 1
    pp_world_size = 2

    engine = launch_engine(
        tp_world_size=tp_world_size,
        pp_world_size=pp_world_size,
        master_host="localhost",
        master_port=29324,
        rpc_port=29123,
        model_fn=vit.create_vit,
    )

    time.sleep(15)  # Wait for engine to start
    asyncio.run(make_requests(10))
