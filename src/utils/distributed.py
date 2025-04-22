import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed(backend='nccl'):
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        gpu = 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(backend=backend,
                          init_method='env://',
                          world_size=world_size,
                          rank=rank)
    
    return rank, world_size, gpu

def cleanup_distributed():
    """Clean up distributed training environment."""
    dist.destroy_process_group()

def get_distributed_sampler(dataset, shuffle=True):
    """Get a distributed sampler for the dataset."""
    return DistributedSampler(dataset, shuffle=shuffle)

def wrap_model(model, device_ids=None):
    """Wrap model with DistributedDataParallel."""
    if device_ids is None:
        device_ids = [torch.cuda.current_device()]
    return DDP(model, device_ids=device_ids)

def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt 