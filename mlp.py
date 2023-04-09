import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, dim, is_pp_first=False, is_pp_last=False):
        super().__init__()
        self.is_pp_first = is_pp_first
        self.is_pp_last = is_pp_last
        layers = [
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        ]
        if is_pp_last:
            layers.append(nn.Linear(dim, 1))
        self.seq = nn.Sequential(*layers)

        self.input_tensor = None

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def forward(self, x):
        if not self.is_pp_first:
            x = self.input_tensor
        return self.seq(x)


def get_model(args, cfg):
    if args.pp_rank == args.num_stages - 1:
        return MLP(cfg.dim, is_pp_last=True)
    else:
        return MLP(cfg.dim)


def forward_step():
    pass


class MLPConfig:
    dim = 32
    train_set_size = 2 ** 12
    eval_set_size = 2 ** 16
    batch_size = 256
    ubatch_size = 16
    n_epochs = 50
    ckpt_epochs = 10
    ckpt_prefix = "mlp"

