import argparse
import asyncio
import logging
import os
import time

from computron import EngineConfig, launch_computron, ModelConfig
from computron.models import opt


engine = None


async def make_request(args, i):
    data = opt.tokenizer("hello world", truncation=True, max_length=512)
    data["max_tokens"] = 1
    data["top_k"] = 50
    data["top_p"] = 0.5
    data["temperature"] = 0.7
    # target = 0
    # target = i // (args.num_requests // 2)
    target = i % args.num_models

    request_time = time.time()
    print(f"{i} request waiting")
    output = await engine.submit(f"opt{target}", data)
    response_time = time.time() - request_time
    if not args.no_log:
        logging.info(f"{i} response time: {response_time}")
    print(f"{i} response time: {response_time}")
    output = opt.tokenizer.decode(output, skip_special_tokens=True)
    print(output)


async def make_blocking_requests(args):
    for i in range(args.num_requests):
        await make_request(args, i)


async def start(args):
    tasks = []
    asyncio.create_task(engine.run())
    start_time = time.time()
    if args.blocking:
        tasks.append(asyncio.create_task(make_blocking_requests(args)))
    else:
        for i in range(args.num_requests):
            tasks.append(asyncio.create_task(make_request(args, i)))
    await asyncio.gather(*tasks)
    print(f"Total time: {time.time() - start_time}")
    await engine.shutdown()


def encode_args(args):
    s = "rr"
    s += f"_{args.model_name}"
    s += f"_n{args.num_models}"
    s += f"_t{args.tp_world_size}"
    s += f"_p{args.pp_world_size}"
    s += f"_r{args.num_requests}"
    if args.blocking:
        s += "_b"
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", default="opt-1.3b")
    parser.add_argument("-n", "--num-models", type=int, default=2)
    parser.add_argument("-t", "--tp-world-size", type=int, default=1)
    parser.add_argument("-p", "--pp-world-size", type=int, default=1)
    parser.add_argument("-r", "--num-requests", type=int, default=24)
    parser.add_argument("-b", "--blocking", action="store_true")
    parser.add_argument("-x", "--no-log", action="store_true")
    args = parser.parse_args()
    print(args)

    if args.no_log:
        log_dir = None
    else:
        log_dir = encode_args(args)
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logging.basicConfig(
            filename=os.path.join(log_dir, "client.log"),
            filemode="w",
            level=logging.INFO,
        )

    # logging.basicConfig(
    #     filename=f"perf_logs/round_robin_{args.model_name}_{args.tp_world_size}_{args.pp_world_size}.log", 
    #     filemode="w",
    #     level=logging.DEBUG
    # )
    # logging.info("==== New Run ====")
    # logging.info("Num models: {}".format(args.num_models))
    # logging.info("TP world size: {}".format(args.tp_world_size))
    # logging.info("PP world size: {}".format(args.pp_world_size))

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
        log_dir=log_dir,
    )

    time.sleep(10)
    asyncio.run(start(args))
