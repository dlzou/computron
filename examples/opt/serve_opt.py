import asyncio
from functools import partial
import time
from pydantic import Field
from computron import launch_multi_model, ModelConfig
import torch
from transformers import GPT2Tokenizer
import opt

ctlr = None

async def make_requests(num_reqs):
    start_time = time.time()
    for i in range(num_reqs):
        data = Field(
        min_length=1, example='Question: Where were the 2004 Olympics held?\nAnswer: Athens, Greece\n\nQuestion: What is the longest river on the earth?\nAnswer:')
        req = opt.OptRequest(data=data)
        target = i % 2
        # target = i // (num_reqs // 2)
        resp = await ctlr.handle_request(f"opt{target}", req)
        print(f"Response time {i}: {time.time() - start_time}")
        # print(resp)
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('model', choices=['opt-125m', 'opt-6.7b', 'opt-30b', 'opt-175b'])  
    # parser.add_argument('--checkpoint', default=None)
    # args = parser.parse_args()

    model_name = 'opt_125M'
    num_models = 2
    tp_world_size = 1
    pp_world_size = 2
    # num_chunks = 1
    first_port = 29600
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    configs = []
    for i in range(num_models):
        config = ModelConfig(
            model_id=f"opt{i}",
            master_host="localhost",
            master_port=(first_port + 3*i),
            rpc_port=(first_port + 3*i + 1),
            request_port=(first_port + 3*i + 2),
            request_type=opt.OptRequest,
            pack_response_fn=opt.pack_response,
            unpack_request_fn=opt.unpack_request,
            model_fn=opt.opt_125M,
            # model_exec_seq=None,
            batch_manager=opt.BatchManagerForGeneration(max_batch_size=1, pad_token_id=opt.tokenizer.pad_token_id),
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

    time.sleep(15) # Wait for engine to start
    asyncio.run(make_requests(10))
