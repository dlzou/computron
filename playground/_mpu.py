import numpy as np
import torch.distributed as dist


_MODEL_WORLD_SIZE = None
_PP_WORLD_SIZE = None
_TP_WORLD_SIZE = None

_MODEL_GROUP = None
_PP_GROUP = None
_TP_GROUP = None


def init_mpu(num_models, pp_size, tp_size, backend):
    assert dist.is_initialized(), "torch.distributed not initialized"

    world_size = dist.get_world_size() - 1
    assert world_size % (num_models * tp_size * pp_size) == 0, "invalid parallelism sizes"
    model_world_size = tp_size * pp_size

    global _MODEL_WORLD_SIZE, _PP_WORLD_SIZE, _TP_WORLD_SIZE
    _MODEL_WORLD_SIZE = model_world_size
    _PP_WORLD_SIZE = pp_size
    _TP_WORLD_SIZE = tp_size

    # Megatron: something about changing mpu sizes on the fly?
    rank = dist.get_rank()
    
    global _MODEL_GROUP, _PP_GROUP, _TP_GROUP
    model_ranks = np.arange(model_world_size)
    for i in range(num_models):
        print(f"model {i} ranks: {model_ranks}")
        # Create model group
        g = dist.new_group(list(model_ranks), backend)
        if rank in model_ranks:
            assert _MODEL_GROUP is None, "model group assigned more than once"
            _MODEL_GROUP = g
        model_ranks += model_world_size
        
        p_cube = np.arange(model_world_size).reshape((pp_size, tp_size))

        # Create pipeline parallel groups
        num_pp_groups = tp_size
        for j in range(num_pp_groups):
            pp_ranks = list(p_cube[:, j])
            print(f"pp {j} ranks: {pp_ranks}")
            g = dist.new_group(pp_ranks, backend)
            if rank in pp_ranks:
                assert _PP_GROUP is None, "PP group assigned more than once"

        # Create tensor parallel groups
        num_tp_groups = pp_size
        for j in range(num_tp_groups):
            tp_ranks = list(p_cube[j, :])
            print(f"tp {j} ranks: {tp_ranks}")
            g = dist.new_group(tp_ranks, backend)
            if rank in tp_ranks:
                assert _TP_GROUP is None, "TP group assigned more than once"


def get_model_group():
    assert _MODEL_GROUP is not None
    return _MODEL_GROUP


def is_pp_first_stage():
    pass


def is_pp_last_stage():
    pass


def get_pp_rank():
    pass


def get_pp_prev_rank():
    pass


def get_pp_next_rank():
    pass