from torch.utils.data import DataLoader, TensorDataset

import os
import sys
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):

    def __init__(self, dim, ubatch_size):
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        ).to("cuda:0")
        self.seq2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        ).to("cuda:1")
        
        self.ubatch_size = ubatch_size
        
    def forward(self, x):
        ubatches = iter(x.split(self.ubatch_size, dim=0))
        ret = []

        # Prologue
        x1 = next(ubatches)
        x2 = self.seq1(x1.to("cuda:0"))

        # Body
        for x1 in ubatches:
            x2 = self.seq2(x2.to("cuda:1"))
            ret.append(x2)

            x2 = self.seq1(x1.to("cuda:0"))

        # Epilogue
        x2 = self.seq2(x2.to("cuda:1"))
        ret.append(x2)

        return torch.cat(ret, dim=0)


dim = 32
train_set_size = 2 ** 12
eval_set_size = 2 ** 16
batch_size = 256
ubatch_size = 16
n_epochs = 50
ckpt_epochs = 10


def train(rank, size):
    model = MLP(dim, ubatch_size)
    model.train()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    inputs = torch.randn(train_set_size, dim)
    targets = torch.zeros(train_set_size, 1) + rank # rank n trains a model that should return n
    train_set = TensorDataset(inputs, targets)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    
    for epoch in range(n_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to("cuda:0"))
            loss_fn(outputs, targets.to(outputs.device)).backward()
            optimizer.step()

        if (epoch + 1) % ckpt_epochs == 0:
            torch.save(model.state_dict(), f"test_{rank}.pt")


def eval(rank, size):
    model = MLP(dim, ubatch_size)
    model.load_state_dict(torch.load(f"test_{rank}.pt"))
    model.eval()

    inputs = torch.randn(eval_set_size, dim)
    eval_set = TensorDataset(inputs)
    eval_loader = DataLoader(eval_set, batch_size=batch_size)

    ret = []
    for inputs in eval_loader:
        outputs = model(inputs[0].to("cuda:0"))
        ret.append(outputs)
    print(torch.mean(torch.cat(ret, dim=0)))


def init_process(rank, size, fn, backend="nccl"):
    """ Initialize the distributed environment. """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    print(f"Starting proc rank={rank}")
    fn(rank, size)

if __name__ == "__main__":
    # for i in range(torch.cuda.device_count()):
    #     print(torch.cuda.get_device_properties(i).name)

    size = int(sys.argv[2])
    processes = []
    mp.set_start_method("spawn")

    if sys.argv[1] == "train":
        fn = train
    elif sys.argv[1] == "eval":
        fn = eval
    else:
        exit(1)

    start_time = time.time()
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, fn))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"Total time: {time.time() - start_time} seconds")