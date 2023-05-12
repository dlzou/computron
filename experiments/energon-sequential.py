import asyncio
import time
import logging
from energonai import launch_engine

import os
import sys

sys.path.append(os.path.abspath("/global/u1/j/jinxch/parallel/Project/cs267-project/examples/opt"))
print(sys.path)
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
        logging.info(str(i)+" server req {} time: {}".format(i, time.time()-start_time))
        engine.submit(uid, inputs)
        output = await engine.wait(uid)
        logging.info(str(i)+" server response {} time: {}".format(i, time.time()-start_time))
        print(f"Response time {i}: {time.time() - start_time}")
        output_seq = tokenizer.decode(output, skip_special_tokens=True)
        print(output_seq)
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":

    logging.basicConfig(filename='logs/energon-sequential.log',filemode='w' ,level=logging.DEBUG)

    tp_world_size = 2
    pp_world_size = 2

    logging.info(" ----   New run ----")
    logging.info("Tp world size: {}".format(tp_world_size))
    logging.info("Pp world size: {}".format(pp_world_size))
    logging.info("Model: {}".format("opt 13B"))

    engine = launch_engine(
        tp_world_size=tp_world_size,
        pp_world_size=pp_world_size,
        master_host="localhost",
        master_port=29600,
        rpc_port=29601,
        model_fn=opt.get_model_fn("opt-13b"),
        batch_manager=opt.BatchManagerForGeneration(
            max_batch_size=4, pad_token_id=tokenizer.pad_token_id
        ),
    )

    time.sleep(10)  # Wait for engine to start



    asyncio.run(make_requests(50))
