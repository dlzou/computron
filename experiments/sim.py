import argparse
import asyncio
from dataclasses import dataclass
import logging
import os
import random
import time

import alpa_serve.simulator.workload as workload
from computron import EngineConfig, ModelConfig, launch_computron
from computron.models import opt 


lock = asyncio.Lock()
req_count = 0
engine = None
args = None
log_fn = None


@dataclass
class Config:
    rates: tuple
    cv: int
    max_loaded: int


configs = [
    Config((1, 1, 1), 0.25, 2),
    Config((1, 1, 1), 1, 2),
    Config((1, 1, 1), 4, 2),
    Config((10, 1, 1), 0.25, 2),
    Config((10, 1, 1), 1, 2),
    Config((10, 1, 1), 4, 2),
    Config((10, 10, 1), 0.25, 2),
    Config((10, 10, 1), 1, 2),
    Config((10, 10, 1), 4, 2),

    Config((1, 1, 1, 1, 1, 1), 0.25, 4),
    Config((1, 1, 1, 1, 1, 1), 1, 4),
    Config((1, 1, 1, 1, 1, 1), 4, 4),
    Config((10, 10, 1, 1, 1, 1), 0.25, 4),
    Config((10, 10, 1, 1, 1, 1), 1, 4),
    Config((10, 10, 1, 1, 1, 1), 4, 4),
    Config((10, 10, 10, 10, 1, 1), 0.25, 4),
    Config((10, 10, 10, 10, 1, 1), 1, 4),
    Config((10, 10, 10, 10, 1, 1), 4, 4),
    # Config((1, 1, 1), 1, 1),
    # Config((10, 10), 1, 1),
    # Config((10, 10, 1), 1, 1),
    # Config((10, 1, 1), 1, 1),
]


async def make_request(index, target):
    data = opt.tokenizer("Berkeley is the number one public", truncation=True, max_length=512)
    data["max_tokens"] = 1
    data["top_k"] = 50
    data["top_p"] = 0.5
    data["temperature"] = 0.7

    request_time = time.time()
    log_fn(f"req{index} model: {target}")
    output = await engine.submit(target, data)
    response_time = time.time() - request_time
    log_fn(f"resp{index} model: {target}, response time: {response_time}")
    output = opt.tokenizer.decode(output, skip_special_tokens=True)
    print(output)


class Client:
    def __init__(self, a, b, model_id, seed=0) -> None:
        global args
        self.arrival = workload.GammaProcess(a, b)
        self.model_id = model_id
        self.request_times = self.arrival.generate_arrivals(0, args.duration, seed)
        log_fn(f"client model: {model_id}, arrival: Gamma({a}, {b}), num requests: {len(self.request_times)}")
    
    async def run(self):
        global lock
        global req_count
        global engine

        # Warm up
        data = opt.tokenizer("hello world", truncation=True, max_length=512)
        data["max_tokens"] = 1
        data["top_k"] = 50
        data["top_p"] = 0.5
        data["temperature"] = 0.7
        await engine.submit(self.model_id, data)

        # Arrival process
        tasks = []
        prev_req_time = 0
        correction_time = time.time()
        for req_time in self.request_times:
            delay = req_time - prev_req_time - (time.time() - correction_time)
            correction_time = time.time()
            prev_req_time = req_time
            await asyncio.sleep(delay)
            async with lock:
                index = req_count
                req_count += 1
            tasks.append(asyncio.create_task(make_request(index, self.model_id)))
        await asyncio.gather(*tasks)


async def start():
    global args
    tasks = []
    asyncio.create_task(engine.run())
    
    config: Config = configs[args.config]
    clients = []
    for i, rate in enumerate(config.rates):
        clients.append(Client(rate, config.cv, f"opt{i}", seed=random.randint(0, 32767)))
    for c in clients:
        tasks.append(asyncio.create_task(c.run()))
    await asyncio.gather(*tasks)
    await engine.shutdown()


def log_nothing(*args):
    pass


def log_and_print(s):
    logging.info(s)
    print(s)


def get_log_dir(args):
    s = "sim"
    s += f"_{args.model_name}"
    s += f"_c{args.config}"
    s += f"_t{args.tp_world_size}"
    s += f"_p{args.pp_world_size}"
    s += f"_b{args.batch_size}"
    s += f"_d{args.duration}"
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", nargs="?", default="opt-1.3b")
    parser.add_argument("-c", "--config", type=int, default=0)
    parser.add_argument("-t", "--tp-world-size", type=int, default=1)
    parser.add_argument("-p", "--pp-world-size", type=int, default=1)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-d", "--duration", type=int, default=30)
    parser.add_argument("-x", "--no-log", action="store_true")
    args = parser.parse_args()
    print(args)

    if args.no_log:
        log_dir = None
        log_fn = log_nothing
    else:
        log_dir = get_log_dir(args)
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            filename=os.path.join(log_dir, "client.log"),
            filemode="w",
            level=logging.INFO,
        )
        log_fn = log_and_print

    config = configs[args.config]
    engine_config = EngineConfig(
        master_host="localhost",
        master_port=29600,
        rpc_port=29601,
        max_loaded=config.max_loaded,
    )
    model_configs = []
    for i in range(len(config.rates)):
        mc = ModelConfig(
            model_id=f"opt{i}",
            model_fn=opt.get_model_fn(args.model_name),
            batch_manager=opt.OPTBatchManager(
                max_batch_size=args.batch_size,
                pad_token_id=opt.tokenizer.pad_token_id,
            ),
        )
        model_configs.append(mc)

    engine = launch_computron(
        engine_config,
        model_configs,
        tp_world_size=args.tp_world_size,
        pp_world_size=args.pp_world_size,
        log_dir=log_dir,
    )

    asyncio.run(start())
