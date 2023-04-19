num_models = 2
num_stages = 4
world_size = num_models * num_stages
local_world_size = 8
local_device_count = 4
assert num_stages == local_device_count * (world_size / local_world_size)

def get_placement(rank):
    local_rank = rank % local_world_size
    node = rank // local_world_size
    model = rank % num_models
    device = local_rank // num_models
    return f"\tlocal_rank={local_rank}\tnode={node}\tmodel={model}\tdevice={device}"

for r in range(world_size):
    print(f"rank {r} placement: " + get_placement(r))
    