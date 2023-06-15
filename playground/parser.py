from os.path import abspath, dirname, join
import re

model_name = "opt-1.3b"
num_models = 2
num_requests = 24

def get_log_dir(tp, pp):
    s = "rr"
    s += f"_{model_name}"
    s += f"_n{num_models}"
    s += f"_t{tp}"
    s += f"_p{tp}"
    s += f"_r{num_requests}"
    s += "_b"
    return join(dirname(abspath(__file__)), s)


def round_robin_times(tp=1, pp=1):
    offload_times = []
    load_times = []
    model_times = []
    total_times = []

    log_dir = get_log_dir(tp, pp)
    with open(join(log_dir, "rank_0_master.log"), "r") as file:
        for line in file:
            if "loaded: False" in line:
                t = re.search(r"time: (\d+\.\d+)", line).group(1)
                offload_times.append(float(t))
            elif "loaded: True" in line:
                t = re.search(r"time: (\d+\.\d+)", line).group(1)
                load_times.append(float(t))
            elif "batch size" in line:
                t = re.search(r"time: (\d+\.\d+)", line).group(1)
                model_times.append(float(t))
    
    with open(join(log_dir, "client.log")) as file:
        for line in file:
            t = re.search(r"time: (\d+\.\d+)", line).group(1)
            total_times.append(float(t))

    # Remove warm up requests
    return (
        offload_times[3:],
        load_times[4:],
        model_times[4:],
        total_times[4:],
    )
