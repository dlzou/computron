import os
import ray
import torch
import torch.nn as nn

import mlp


@ray.remote(num_gpus=0.5)
class ModelWorker:
    def __init__(self, model: nn.Module):
        self.model = model
    
    def handle_request(self, request):
        x = request
        self.model.set_input_tensor(x)
        return self.model(x)
    
    def get_device(self):
        print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


if __name__ == "__main__":
    ray.init(num_gpus=1)
    m1 = mlp.MLP(32, is_pp_last=True)
    m2 = mlp.MLP(32, is_pp_last=True)
    w1 = ModelWorker.remote(m1)
    w2 = ModelWorker.remote(m2)

    print(ray.get(w1.get_device.remote()))
    print(ray.get(w2.get_device.remote()))
    print(ray.get(w1.handle_request.remote(torch.ones(1, 32))))
    print(ray.get(w2.handle_request.remote(torch.ones(1, 32))))