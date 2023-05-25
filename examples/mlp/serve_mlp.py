import argparse
import asyncio
from functools import partial
import time

from computron import EngineConfig, launch_computron, ModelConfig
from computron.models import mlp
import torch


engine = None


async def make_requests(num_reqs, dim):
    start_time = time.time()
    for i in range(num_reqs):
        data = torch.ones((dim,)) * i
        # target = 0
        # target = i % 2
        target = i // (num_reqs // 2)
        print(f"Making request {i}")
        request_time = time.time()
        output = await engine.submit(f"mlp{target}", data)
        print(f"Response time {i}: {time.time() - request_time}")
        print(torch.mean(output))
    print(f"Total time: {time.time() - start_time}")


async def start(args):
    engine_task = asyncio.create_task(engine.run())
    request_task = asyncio.create_task(make_requests(args.num_requests, args.dim))
    await asyncio.gather(engine_task, request_task)
    await engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dim", type=int, default=256)
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
            model_id=f"mlp{i}",
            model_fn=partial(mlp.MLP, offset=10*i, dim=args.dim),
            pipelinable=True,
            batch_manager=mlp.MLPBatchManager(max_batch_size=4),
        )
        model_configs.append(mc)
    

    engine = launch_computron(
        engine_config,
        model_configs,
        tp_world_size=args.tp_world_size,
        pp_world_size=args.pp_world_size,
    )

    asyncio.run(start(args))
