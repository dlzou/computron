import asyncio
import time

from computron import launch_multi_model, ModelConfig

import echo


ctlr = None


async def make_requests(num_reqs):
    start_time = time.time()
    for i in range(num_reqs):
        req = echo.EchoRequest(data=f"hello world {i}")
        target = i % 2
        # target = i // (num_reqs // 2)
        resp = await ctlr.handle_request(f"echo{target}", req)
        print(f"Response time {i}: {time.time() - start_time}")
        print(resp)
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    num_models = 2
    first_port = 29600
    configs = []
    for i in range(num_models):
        config = ModelConfig(
            model_id=f"echo{i}",
            master_host="localhost",
            master_port=(first_port + 3*i),
            rpc_port=(first_port + 3*i + 1),
            request_port=(first_port + 3*i + 2),
            request_type=echo.EchoRequest,
            unpack_request_fn=echo.unpack_request,
            pack_response_fn=echo.pack_response,
            model_fn=echo.Echo,
        )
        configs.append(config)

    ctlr = launch_multi_model(
        configs,
        tp_world_size=1,
        pp_world_size=1,
        n_nodes=1,
        node_rank=0,
        controller_kwargs={
            "max_loaded": 1,
        },
    )

    time.sleep(5)
    asyncio.run(make_requests(10))