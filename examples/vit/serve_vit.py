import asyncio
from functools import partial
import time
from proc_img import proc_img
from computron import launch_multi_model, ModelConfig
import torch

import vit

ctlr = None


async def make_requests(num_reqs):
    start_time = time.time()
    for i in range(num_reqs):
        img = proc_img('/data/yusun/xueyang/cs267/cs267-project/examples/vit/dataset/n01667114_9985.JPEG')    
        req = vit.vitRequest(data=img)
        target = i % 2
        # target = i // (num_reqs // 2)
        resp: vit.vitResponse = await ctlr.handle_request(f"vit{target}", req)
        print(f"Response time {i}: {time.time() - start_time}")
        print(resp.output.shape)
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    num_models = 2
    tp_world_size = 1
    pp_world_size = 2
    first_port = 29234
    # dim = 256

    configs = []
    for i in range(num_models):
        config = ModelConfig(
            model_id=f"vit{i}",
            master_host="localhost",
            master_port=(first_port + 3*i),
            rpc_port=(first_port + 3*i + 1),
            request_port=(first_port + 3*i + 2),
            request_type=vit.vitRequest,
            unpack_request_fn=vit.unpack_request,
            pack_response_fn=vit.pack_response,
            model_fn=vit.create_vit,
            # pipelinable=True,
            batch_manager=vit.vitBatchManager(max_batch_size=1),
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

    time.sleep(20) # Wait for engine to start
    asyncio.run(make_requests(10))
