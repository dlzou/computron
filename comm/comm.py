import requests
import time
import threading
import torch
import os
import pickle
import sys
print(os.path.abspath(os.path.curdir))
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../cs267-project"))
print(sys.path)

import alpa_serve.simulator.workload as workload

import socket

HOST = 'localhost'
PORT = 5000


class Server:
    def __init__(self):
        self.sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((HOST,PORT))
        self.responseid=0
    
    def bind(self, engine):
        self.engine=engine
    def monitor(self):
        while True:
            data_bytes, addr = self.sock.recvfrom(10240)
            data=pickle.loads(data_bytes)
            print('Received:', data)
            self.sock.sendto(b'Received', addr)
            print("Response id: ",self.responseid)
            self.responseid+=1
            uid=id(data)
            self.engine.submit(uid, data)
            print("Submmitted: ",uid)


class Client:
    def __init__(self, a, b) -> None:
        self.process=workload.GammaProcess(a,b)
        self.url="localhost:1234"
        self.sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        
    def gen(self, st,duration,seed=0):
        self.request_time=self.process.generate_arrivals(st,duration,seed)    
        print(self.request_time)

    def start(self):
        start_time=0
        ptime=time.time()
        ctime=time.time()
        for time_point in self.request_time:
            delay = time_point - start_time
            start_time=time_point
            print(delay)
            delay-=time.time()-ctime
            time.sleep(delay)
            ctime=time.time()
            print("current request time: ",time.time()-ptime)

            #send
            data = torch.ones((256,))
            data_bytes=pickle.dumps(data)
            self.sock.sendto(data_bytes,(HOST,PORT))
            response,addr=self.sock.recvfrom(1024)
            
            print(f"Response status code: {response.decode()}")



import asyncio
from functools import partial
import time

from energonai import launch_engine
import torch
import src.models.mlp as mlp

if __name__=="__main__":

    tp_world_size = 2
    pp_world_size = 1
    
    engine = launch_engine(
        tp_world_size=tp_world_size,
        pp_world_size=pp_world_size,
        master_host="localhost",
        master_port=29600,
        rpc_port=29601,
        model_fn=partial(mlp.MLP, dim=256),
    )

    time.sleep(15) # Wait for engine to start

    client=Client(1,2)
    client.gen(0,10)

    server=Server()
    server.bind(engine)


    client_thread= threading.Thread(target=client.start)
    server_thread= threading.Thread(target=server.monitor)

    server_thread.start()
    client_thread.start()
