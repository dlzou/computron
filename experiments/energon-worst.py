import asyncio
import time

from energonai import launch_engine

import threading

import logging

import os
import sys
sys.path.append(os.path.abspath("/global/u1/j/jinxch/parallel/Project/cs267-project/examples/opt"))
print(sys.path)
import opt


engine = None
tokenizer = opt.tokenizer

uids=[]
thrds=[]
start_time=None

async def get_res(id):
    # print(id)
    # print("STARTED")
    output = await engine.wait(uids[id])
    logging.info(str(id)+" response time: {}".format(time.time()-start_time))
    print(f"Response time {id}: {time.time() - start_time}")
    output_seq = tokenizer.decode(output, skip_special_tokens=True)
    print(output_seq)

async def make_requests(num_reqs):
    global start_time
    start_time = time.time()
    
    tasks=[]
    
    for i in range(num_reqs):
        task=asyncio.create_task(get_res(i))
        tasks.append(task)

    for i in range(num_reqs):
        inputs = tokenizer("hello world", truncation=True, max_length=512)
        inputs["max_tokens"] = 1
        inputs["top_k"] = 50
        inputs["top_p"] = 0.5
        inputs["temperature"] = 0.7

        logging.info(str(i)+" req time: {}".format(time.time()-start_time))

        uid = id(inputs)
        # print(uid)
        engine.submit(uid, inputs)
        uids.append(uid)

    logging.info(time.time()-start_time)
    
    await asyncio.wait(tasks)

    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":

    logging.basicConfig(filename='logs/energon-worst-temp.log', level=logging.DEBUG)


    tp_world_size = 1
    pp_world_size = 2

    logging.info("\nNew run --- ")
    logging.info("Tp world size: {}".format(tp_world_size))
    logging.info("Pp world size: {}".format(pp_world_size))

    engine = launch_engine(
        tp_world_size=tp_world_size,
        pp_world_size=pp_world_size,
        master_host="localhost",
        master_port=29600,
        rpc_port=29601,
        model_fn=opt.get_model_fn("opt-6.7b"),
        batch_manager=opt.BatchManagerForGeneration(
            max_batch_size=4, pad_token_id=tokenizer.pad_token_id
        ),
    )

    time.sleep(10)  # Wait for engine to start
    asyncio.run(make_requests(20))
