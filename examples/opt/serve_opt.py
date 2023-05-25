import argparse
import asyncio
import time

from computron import EngineConfig, launch_computron, ModelConfig
from computron.models import opt


engine = None


async def make_requests(num_reqs):
    start_time = time.time()
    for i in range(num_reqs):
        data = opt.tokenizer("hello world", truncation=True, max_length=512)
        data["max_tokens"] = 1
        data["top_k"] = 50
        data["top_p"] = 0.5
        data["temperature"] = 0.7
        # target = 0
        # target = i // (num_reqs // 2)
        target = i % 2
        print(f"Making request {i}")
        req_time = time.time()
        output = await engine.submit(f"opt{target}", data)
        print(f"Response time {i}: {time.time() - req_time}")
        output = opt.tokenizer.decode(output, skip_special_tokens=True)
        print(output)
    print(f"Total time: {time.time() - start_time}")


async def start(args):
    engine_task = asyncio.create_task(engine.run())
    request_task = asyncio.create_task(make_requests(args.num_requests))
    await asyncio.gather(engine_task, request_task)
    # await engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", default="opt-1.3b")
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
            model_id=f"opt{i}",
            model_fn=opt.get_model_fn(args.model_name),
            batch_manager=opt.OPTBatchManager(
                max_batch_size=4, pad_token_id=opt.tokenizer.pad_token_id
            ),
        )
        model_configs.append(mc)

    engine = launch_computron(
        engine_config,
        model_configs,
        tp_world_size=args.tp_world_size,
        pp_world_size=args.pp_world_size,
        # log_dir="logs",
    )

    asyncio.run(start(args))
