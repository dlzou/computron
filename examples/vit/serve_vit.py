import argparse
import asyncio
import os
import time

from computron import EngineConfig, ModelConfig, launch_computron
from computron.models import vit

from proc_img import proc_img


engine = None


async def make_requests(num_reqs):
    start_time = time.time()
    for i in range(num_reqs):
        img = proc_img(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "dataset/n01667114_9985.JPEG",
            )
        )
        # target = 0
        # target = i // (num_reqs // 2)
        target = i % 2
        print(f"Making request {i}")
        req_time = time.time()
        output = await engine.submit(f"vit{target}", img)
        print(f"Response time {i}: {time.time() - req_time}")
        print(output.shape)
    print(f"Total time: {time.time() - start_time}")


async def start(args):
    engine_task = asyncio.create_task(engine.run())
    request_task = asyncio.create_task(make_requests(args.num_requests))
    await asyncio.gather(engine_task, request_task)
    await engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-models", type=int, default=2)
    parser.add_argument("-t", "--tp-world-size", type=int, default=1)
    parser.add_argument("-p", "--pp-world-size", type=int, default=1)
    parser.add_argument("-r", "--num-requests", type=int, default=12)
    args = parser.parse_args()
    print(args)
    
    engine_config = EngineConfig(
        master_host="localhost",
        master_port=29600,
        rpc_port=29601,
        max_loaded=1,
    )

    model_configs = []
    for i in range(args.num_models):
        mc = ModelConfig(
            model_id=f"vit{i}",
            model_fn=vit.create_vit,
            batch_manager=vit.ViTBatchManager(max_batch_size=4),
        )
        model_configs.append(mc)

    engine = launch_computron(
        engine_config,
        model_configs,
        tp_world_size=args.tp_world_size,
        pp_world_size=args.pp_world_size,
    )

    asyncio.run(start(args))
