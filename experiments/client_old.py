import requests
import time

import os
import sys

print(os.path.abspath(os.path.curdir))
sys.path.append(os.path.abspath("."))
print(sys.path)
import alpa_serve.simulator.workload as workload


class Client:
    def __init__(self, a, b) -> None:
        self.process = workload.GammaProcess(a, b)
        self.url = "localhost:1234"

    def gen(self, st, duration, seed=0):
        self.request_time = self.process.generate_arrivals(st, duration, seed)
        print(self.request_time)

    def start(self):
        start_time = 0
        ptime = time.time()
        ctime = time.time()
        for time_point in self.request_time:
            delay = time_point - start_time
            start_time = time_point
            print(delay)
            delay -= time.time() - ctime
            time.sleep(delay)
            ctime = time.time()
            print("request: ", time.time() - ptime)
            response = requests.get(self.url)

            print(f"Response status code: {response.status_code}")


client = Client(1, 2)
client.gen(0, 10)
client.start()
