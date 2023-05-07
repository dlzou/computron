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

sys.path.append(os.path.abspath("/global/u1/j/jinxch/parallel/Project/cs267-project/examples/opt"))
import opt

sys.path.append(os.path.abspath("/global/u1/j/jinxch/parallel/Project/cs267-project/alpa"))
import alpa_serve.simulator.workload as workload
import socket


msg_queue = None


controller = None
start_time = None

async def get_res(i,target):
    print(start_time)
    print(i)

    req = opt.OPTRequest(max_tokens=1, prompt="hello world")
    # target = 0
    # target = i // (num_reqs // 2)

    logging.info(str(i)+" server req time: {}".format(time.time()-start_time))

    try:
        resp: opt.OPTResponse = await controller.handle_request(f"opt{target}", req)
    except Exception as e:
        print(e)

    logging.info(str(i)+" server response time: {}".format(time.time()-start_time))

    print(f"Response time {i}: {time.time() - start_time}")
    print(resp.output)

# async def make_requests(num_reqs):
#     global start_time
#     start_time = time.time()

#     tasks=[]
#     for i in range(num_reqs):
#         task=asyncio.create_task(get_res(i))
#         tasks.append(task)

#     logging.info(time.time()-start_time)
    
#     await asyncio.wait(tasks)

#     print(f"Total time: {time.time() - start_time}")


def serve():

    def monitor(self):
        while True:
            # data_bytes, addr = self.sock.recvfrom(10240)
            if msg_queue.empty():
                continue
            sender_id, data_bytes = msg_queue.get()
            data = pickle.loads(data_bytes)
            print("Received from {}: {}".format(sender_id, data))

            # self.sock.sendto(b"Received", addr)

            # AWAIT?

            print("Response id: ", self.responseid)
            self.responseid += 1
            uid = id(data)
            self.engine.submit(uid, data)
            print("Submmitted: ", uid)


class Client:
    def __init__(self, a, b, model_id, st_id) -> None:
        self.process = workload.GammaProcess(a, b)
        self.url = "localhost:1234"
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.model_id = model_id

    def gen(self, st, duration, seed=0):
        self.request_time = self.process.generate_arrivals(st, duration, seed)
        print(self.request_time)
        logging.info("client {}, req len: {}".format(self.cnt, len(self.request_time)))
        for i in self.request_time:
            logging.info(i)

    def start(self):
        start_time = 0
        ptime = time.time()
        ctime = time.time()
        for time_point in self.request_time:
            delay = time_point - start_time
            start_time = time_point
            print(delay)
            delay -= time.time() - ctime
            if delay>0:
                time.sleep(delay)
            ctime = time.time()
            print("current request time: ", time.time() - ptime)

            #  send from client to server (a request)
            data = torch.ones((256,))
            data_bytes = pickle.dumps(data)

            msg_queue.put((self.id, data_bytes))

            global global_cnt
            
            msg_queue.put((global_cnt, self.model_id))

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

if __name__ == "__main__":

    logging.basicConfig(filename='logs/gamma.log', level=logging.DEBUG)


    num_models = 2
    tp_world_size = 1
    pp_world_size = 2

    logging.info("\nNew run --- ")
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

    time.sleep(15)  # Wait for engine to start

    msg_queue = queue.Queue()

    clients = []
    clients.append(Client(1, 2, 1))
    clients.append(Client(1, 2, 2))
    # client = Client(1, 2, 1)
    for i in range(len(clients)):
        clients[i].gen(0, 10, seed=random.randint(0,32767))

    start_time=time.time()
    print("1234214",start_time)

    client_threads = [None] * 2
    for i in range(len(clients)):
        threads.append(threading.Thread(target=clients[i].start))
    threads.append(threading.Thread(target=serve))

    for i in range(len(threads)):
        threads[i].start()

    for i in range(len(threads)):
        threads[i].join()


    

