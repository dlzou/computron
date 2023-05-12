import argparse
import asyncio
import time


import logging

from computron import launch_multi_model, ModelConfig

# TODO: package example models
import sys
import os
from os.path import dirname

opt_path = os.path.join(dirname(dirname(__file__)), "examples/opt")
sys.path.append(opt_path)
import opt


controller = None


async def make_requests(num_models, num_requests):
    start_time = time.time()
    for i in range(num_requests):
        req = opt.OPTRequest(max_tokens=1, prompt="hello world")
        # target = 0
        # target = i // (num_reqs // 2)
        target = i % num_models

        request_time = time.time()

        resp, timers = await controller.handle_request(f"opt{target}", req)
        
        logging.info(f"{i} load time: {timers['load']}")
        logging.info(f"{i} model time: {timers['model']}")
        logging.info(f"{i} response time: {time.time() - request_time}")

        print(f"Response time {i}: {time.time() - request_time}")
        print(resp.output)
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", default="opt-1.3b")
    parser.add_argument("-n", "--num-models", type=int, default=2)
    parser.add_argument("-t", "--tp-world-size", type=int, default=1)
    parser.add_argument("-p", "--pp-world-size", type=int, default=1)
    parser.add_argument("-r", "--num-requests", type=int, default=24)
    args = parser.parse_args()
    print(args)

    logging.basicConfig(
        filename=f"perf_logs/round_robin_{args.model_name}_{args.tp_world_size}_{args.pp_world_size}.log", 
        filemode="w",
        level=logging.DEBUG
    )
    logging.info("==== New Run ====")
    logging.info("Num models: {}".format(args.num_models))
    logging.info("TP world size: {}".format(args.tp_world_size))
    logging.info("PP world size: {}".format(args.pp_world_size))

    first_port = 29600
    configs = []
    for i in range(args.num_models):
        config = ModelConfig(
            model_id=f"opt{i}",
            master_host="localhost",
            master_port=(first_port + 3 * i),
            rpc_port=(first_port + 3 * i + 1),
            request_port=(first_port + 3 * i + 2),
            request_type=opt.OPTRequest,
            unpack_request_fn=opt.unpack_request,
            pack_response_fn=opt.pack_response,
            model_fn=opt.get_model_fn(args.model_name),
            batch_manager=opt.OPTBatchManager(
                max_batch_size=4, pad_token_id=opt.tokenizer.pad_token_id
            ),
        )
        configs.append(config)

    controller = launch_multi_model(
        configs,
        tp_world_size=args.tp_world_size,
        pp_world_size=args.pp_world_size,
        n_nodes=1,
        node_rank=0,
        controller_kwargs={
            "max_loaded": args.num_models - 1,
        },
        # log_dir="logs",
    )

    asyncio.run(make_requests(args.num_models, args.num_requests))
