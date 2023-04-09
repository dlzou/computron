
import math
import torch
import torch.distributed as dist

import mpu


def pipeline_forward(args, cfg, batch):
    num_ubatches = math.ceil(cfg.batch_size / cfg.ubatch_size)
    ubatches = (batch[b:b+cfg.ubatch_size] for b in range(len(batch), cfg.ubatch_size))

    # Warmup passes
    num_warmup_ubatches = args.pp_world_size - args.pp_rank - 1
    num_warmup_ubatches = min(num_warmup_ubatches, num_ubatches)
    num_ubatches_remaining = num_ubatches - num_warmup_ubatches

    for i in range(num_warmup_ubatches):
        pass
    
    for i in range(num_ubatches_remaining):
        pass
