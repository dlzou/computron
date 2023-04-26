import asyncio
from functools import partial
import time

from energonai import launch_engine
from energonai.model import opt_125M, opt_6B, opt_30B

import opt 


engine = None
tokenizer = opt.tokenizer


async def make_requests(num_reqs):
    start_time = time.time()
    for i in range(num_reqs):
        inputs = tokenizer("hello world", truncation=True, max_length=512)
        inputs["max_tokens"] = 1
        inputs["top_k"] = 50
        inputs["top_p"] = 0.5
        inputs["temperature"] = 0.7

        uid = id(inputs)
        engine.submit(uid, inputs)
        output = await engine.wait(uid)
        print(f"Response time {i}: {time.time() - start_time}")
        output = tokenizer.decode(output, skip_special_tokens=True)
        print(output)
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    tp_world_size = 2
    pp_world_size = 1
    
    engine = launch_engine(
        tp_world_size=tp_world_size,
        pp_world_size=pp_world_size,
        master_host="localhost",
        master_port=29600,
        rpc_port=29601,
        model_fn=opt.get_model_fn("opt-125m"),
        batch_manager=opt.BatchManagerForGeneration(
            max_batch_size=5,
            pad_token_id=tokenizer.pad_token_id
        ),
    )

    time.sleep(15) # Wait for engine to start
    asyncio.run(make_requests(10))
