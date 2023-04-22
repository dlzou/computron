import asyncio
from functools import partial
import time

import torch

from launch import launch_multi_model, ModelConfig
from models import mlp


ctlr = None


async def make_requests(num_reqs):
    start_time = time.time()
    for i in range(num_reqs):
        data = torch.ones((32,)) * i
        req = mlp.MLPRequest(data=data)
        resp = await ctlr.handle_request(f"mlp{i % 2}", req)
        print(f"Response time {i}: {time.time() - start_time}")
        print(resp)
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    num_models = 2
    tp_world_size = 1
    pp_world_size = 2
    num_chunks = 1
    first_port = 29600

    configs = []
    for i in range(num_models):
        config = ModelConfig(
            model_id=f"mlp{i}",
            master_host="localhost",
            master_port=(first_port + 3*i),
            rpc_port=(first_port + 3*i + 1),
            request_port=(first_port + 3*i + 2),
            request_type=mlp.MLPRequest,
            unpack_request_fn=mlp.unpack_request,
            pack_response_fn=mlp.pack_response,
            model_fn=partial(mlp.MLP, dim=32),
            model_exec_seq=mlp.exec_seq,
        )
        configs.append(config)

    ctlr = launch_multi_model(
        configs,
        tp_world_size=tp_world_size,
        pp_world_size=pp_world_size,
        n_nodes=1,
        node_rank=0,
    )

    time.sleep(15) # Wait for engine to start
    asyncio.run(make_requests(6))
