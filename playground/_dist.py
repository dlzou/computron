from torch.utils.data import DataLoader, TensorDataset

import argparse
import os
import time
import torch


def eval(args, module, cfg):
    print(f"Starting eval() on rank {args.rank}")
    model = module.MLP(cfg.dim, is_pp_last=True)
    # model.load_state_dict(torch.load(config.ckpt_prefix + f"_{args.rank}.pt"))
    model.to(args.device)
    model.eval()

    # TODO: per-model dataset
    inputs = torch.randn(cfg.eval_set_size, cfg.dim)
    eval_set = TensorDataset(inputs)
    eval_loader = DataLoader(eval_set, batch_size=cfg.batch_size)

    ret = []
    for in_batch in eval_loader:
        model.set_input_tensor(in_batch[0].to(args.device))
        out_batch = model(in_batch[0])
        ret.append(out_batch)
    print(torch.mean(torch.cat(ret, dim=0)) + args.rank * 10)
    

def run(model):
    pass


def main():
    # Run multiple colocated models with pipeline parallelism (no data nor tensor parallelism).

    # Example with 2 models, 2 stages, 2 devices:
    # World size: 8
    # Model processes: m0 = [0, 1], m1 = [2, 3]
    # Device placement: [0, 2], [1, 3]

    parser = argparse.ArgumentParser(prog="dist_test", description="Testing distributed pipeline parallelism")
    parser.add_argument("action", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("-m", "--num-models", type=int, required=True)
    # parser.add_argument("-p", "--pp-world-size", type=int, required=True)
    
    args = parser.parse_args()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    

    args.world_size = int(os.environ["WORLD_SIZE"])
    # assert args.world_size == args.num_models * args.pp_world_size, \
    #     "expected world_size == num_models * pp_world_size"
    args.rank = int(os.environ["RANK"])

    # Parallel state
    # local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    # local_rank = int(os.environ["LOCAL_RANK"])
    # assert local_rank == args.rank % local_world_size

    # local_device_count = torch.cuda.device_count() 
    # assert args.pp_world_size == local_device_count * (args.world_size / local_world_size)
    
    # args.node_id = args.rank // local_world_size
    # args.device_id = local_rank // args.num_models
    # args.model_id = args.rank % args.num_models
    # args.pp_rank = args.rank // args.num_models
    # assert 0 <= args.pp_rank and args.pp_rank < args.pp_world_size
    
    # torch.cuda.set_device(args.device_id)
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    args.device = torch.device("cuda:0")

    if args.model == "mlp":
        import mlp
        module = mlp
        config = mlp.MLPConfig
    else:
        exit(1)


    if args.action == "train":
        # train(args, module, config)
        exit(1)
    elif args.action == "eval":
        eval(args, module, config)
    else:
        exit(1)


if __name__ == "__main__":
    main()