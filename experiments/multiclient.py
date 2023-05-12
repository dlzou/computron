import requests
import time
import threading
import torch
import os
import pickle
import sys
import queue
import logging

lock=threading.RLock()
global_cnt=0

from computron import launch_multi_model, ModelConfig

from os.path import dirname
opt_path = os.path.join(dirname(dirname(__file__)), "examples/opt")
sys.path.append(opt_path)
import opt

opt_path = os.path.join(dirname(dirname(__file__)), "alpa")
sys.path.append(opt_path)
import alpa_serve.simulator.workload as workload
import socket


msg_queue = None


controller = None
start_time = None
num_models = None

# async def get_res(i,target):
#     print(start_time)
#     print(i)

#     req = opt.OPTRequest(max_tokens=1, prompt="hello world")
#     # target = 0
#     # target = i // (num_reqs // 2)

#     logging.info(str(i)+" server req time: {}".format(time.time()-start_time))

#     try:
#         task=asyncio.create_task(controller.handle_request(f"opt{target}", req))
#         resp: opt.OPTResponse = await task
#     except Exception as e:
#         print(e)

#     logging.info(str(i)+" server response time: {}".format(time.time()-start_time))

#     print(f"Response time {i}: {time.time() - start_time}")
#     print(resp.output)

async def worker():
    ### warmup

    req = opt.OPTRequest(max_tokens=1, prompt="hello world")
    for i in range(num_models):
        print("warm up",num_models)
        resp: opt.OPTResponse = await controller.handle_request(f"opt{i}", req)
    while True:
        await asyncio.sleep(0.01)
        # print("hello", msg_queue.empty())
        if not msg_queue.empty():
            req_id, model_id = await msg_queue.get()
            print("Server request", req_id,model_id)
            req = opt.OPTRequest(max_tokens=1, prompt="hello world")
            # target = 0
            # target = i // (num_reqs // 2)

            logging.info(str(req_id)+" server req {} time: {}".format(req_id, time.time()-start_time))

            try:
                resp: opt.OPTResponse = await controller.handle_request(f"opt{model_id}", req)
            except Exception as e:
                print(e)

            logging.info(str(req_id)+" server response {} time: {}".format(req_id, time.time()-start_time))

            print(f"Server Response time {req_id}: {time.time() - start_time}")
            print(resp.output)
        


        


class Client:
    def __init__(self, a, b, model_id) -> None:
        self.process = workload.GammaProcess(a, b)
        self.url = "localhost:1234"
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.model_id = model_id

    def gen(self, st, duration, seed=0):
        self.request_time = self.process.generate_arrivals(st, duration, seed)
        print(self.request_time)
        logging.info("client {}, req len: {}".format(self.model_id, len(self.request_time)))
        for i in self.request_time:
            logging.info(i)

    async def start(self):
        global global_cnt
        sttime = 0
        ptime = time.time()
        ctime = time.time()
        for time_point in self.request_time:
            delay = time_point - sttime
            sttime = time_point
            print(delay)
            delay -= time.time() - ctime
            if delay>0:
                await asyncio.sleep(delay)
            ctime = time.time()
            print("current request time: ", time.time() - ptime)

            logging.info("client {}, req {} time: {}".format(self.model_id, global_cnt, time.time() - start_time))

            #  send from client to server (a request)
            # data = torch.ones((256,))
            # data_bytes = pickle.dumps(data)

            # msg_queue.put((self.id, data_bytes))

           
            
            await msg_queue.put((global_cnt, self.model_id))

            # asyncio.run(get_res(self.cnt,self.model_id))
            
            with lock:
                global_cnt+=1


        # msg_queue.put((999,999))


import asyncio
from functools import partial
import time

from energonai import launch_engine
import torch
import random

async def main_start():
    global start_time
    tasks = [None] * len(clients)
    for i in range(len(clients)):
        tasks[i]=(asyncio.create_task(clients[i].start()))

    num_worker=1
    for i in range(num_worker):
        tasks.append(asyncio.create_task(worker()))

    start_time=time.time()
    await asyncio.wait(tasks)

if __name__ == "__main__":

    logging.basicConfig(filename='logs/gamma.log',filemode='w' ,level=logging.DEBUG)


    num_models = 3
    tp_world_size = 2   
    pp_world_size = 2
    maxload=2
    logging.info(" ----   New run ----")
    logging.info("Num models:{}".format(num_models))
    logging.info("Tp world size: {}".format(tp_world_size))
    logging.info("Pp world size: {}".format(pp_world_size))
    logging.info("Max load: {}".format(maxload))
    logging.info("Model: {}".format("opt 30B"))

    first_port = 29700



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
            model_fn=opt.opt_6B,
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
            "max_loaded": maxload,
        },
        # log_dir="logs",
    )

    # time.sleep(15)  # Wait for engine to start

    msg_queue = asyncio.Queue()

    clients = []
    clients.append(Client(1, 2, 0)) #0 is model id
    clients.append(Client(1, 2, 1))
    clients.append(Client(1, 2, 2))
    # client = Client(1, 2, 1)
    for i in range(len(clients)):
        clients[i].gen(0, 25, seed=random.randint(0,32767))

    start_time=time.time()
    # print("1234214",start_time)

   
    asyncio.run(main_start())

    

