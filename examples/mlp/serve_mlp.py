import asyncio
from functools import partial
import time

from computron import launch_multi_model, ModelConfig
import torch

import mlp

ctlr = None


async def make_requests(num_reqs, dim):
    start_time = time.time()
    for i in range(num_reqs):
        data = torch.ones((dim,)) * i
        req = mlp.MLPRequest(data=data)
        target = i % 2
        # target = i // (num_reqs // 2)
        resp: mlp.MLPResponse = await ctlr.handle_request(f"mlp{target}", req)
        print(f"Response time {i}: {time.time() - start_time}")
        print(resp.output.shape)
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    num_models = 2
    tp_world_size = 2
    pp_world_size = 1
    first_port = 29600
    dim = 256

    configs = []
    for i in range(num_models):
        config = ModelConfig(
            model_id=f"mlp{i}",
            master_host="localhost",
            master_port=(first_port + 3 * i),
            rpc_port=(first_port + 3 * i + 1),
            request_port=(first_port + 3 * i + 2),
            request_type=mlp.MLPRequest,
            unpack_request_fn=mlp.unpack_request,
            pack_response_fn=mlp.pack_response,
            model_fn=partial(mlp.MLP, dim=dim),
            pipelinable=True,
            batch_manager=mlp.MLPBatchManager(max_batch_size=1),
        )
        configs.append(config)

    ctlr = launch_multi_model(
        configs,
        tp_world_size=tp_world_size,
        pp_world_size=pp_world_size,
        n_nodes=1,
        node_rank=0,
        controller_kwargs={
            "max_loaded": 1,
        },
    )

    time.sleep(20)  # Wait for engine to start
    asyncio.run(make_requests(10, dim))
