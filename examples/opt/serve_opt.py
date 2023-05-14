import argparse
import asyncio
import time

from computron import launch_multi_model, ModelConfig
import opt


controller = None


async def make_requests(num_reqs):
    start_time = time.time()
    for i in range(num_reqs):
        req = opt.OPTRequest(max_tokens=1, prompt="hello world")
        # target = 0
        # target = i // (num_reqs // 2)
        target = i % 2
        req_time = time.time()
        resp, _ = await controller.handle_request(f"opt{target}", req)
        print(f"Response time {i}: {time.time() - req_time}")
        print(resp.output)
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", default="opt-1.3b")
    parser.add_argument("-n", "--num-models", type=int, default=2)
    parser.add_argument("-t", "--tp-world-size", type=int, default=1)
    parser.add_argument("-p", "--pp-world-size", type=int, default=1)
    parser.add_argument("-r", "--num-requests", type=int, default=12)
    args = parser.parse_args()
    print(args)
    
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
            batch_manager=opt.BatchManagerForGeneration(
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
            "max_loaded": 1,
        },
        # log_dir="logs",
    )

    asyncio.run(make_requests(args.num_requests))
