import asyncio
import time

import logging

from computron import launch_multi_model, ModelConfig

import sys
import os
sys.path.append(os.path.abspath("/global/u1/j/jinxch/parallel/Project/cs267-project/examples/opt"))

import opt


controller = None


async def make_requests(num_reqs):
    start_time = time.time()
    for i in range(num_reqs):
        req = opt.OPTRequest(max_tokens=1, prompt="hello world")
        # target = 0
        # target = i // (num_reqs // 2)
        target = i % 2

        logging.info(str(i)+" req time: {}".format(time.time()-start_time))

        resp: opt.OPTResponse = await controller.handle_request(f"opt{target}", req)

        logging.info(str(i)+" response time: {}".format(time.time()-start_time))

        print(f"Response time {i}: {time.time() - start_time}")
        print(resp.output)
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":

    logging.basicConfig(filename='logs/worst.log', level=logging.DEBUG)


    num_models = 2
    tp_world_size = 1
    pp_world_size = 2

    logging.info("New run --- ")
    logging.info("Num models:{}".format(num_models))
    logging.info("Tp world size: {}".format(tp_world_size))
    logging.info("Pp world size: {}".format(pp_world_size))

    first_port = 29600



    configs = []
    for i in range(num_models):
        config = ModelConfig(
            model_id=f"opt{i}",
            master_host="localhost",
            master_port=(first_port + 3 * i),
            rpc_port=(first_port + 3 * i + 1),
            request_port=(first_port + 3 * i + 2),
            request_type=opt.OPTRequest,
            unpack_request_fn=opt.unpack_request,
            pack_response_fn=opt.pack_response,
            model_fn=opt.opt_1B,
            batch_manager=opt.OPTBatchManager(
                max_batch_size=4, pad_token_id=opt.tokenizer.pad_token_id
            ),
        )
        configs.append(config)

    controller = launch_multi_model(
        configs,
        tp_world_size=tp_world_size,
        pp_world_size=pp_world_size,
        n_nodes=1,
        node_rank=0,
        controller_kwargs={
            "max_loaded": 1,
        },
        # log_dir="logs",
    )

    asyncio.run(make_requests(20))
