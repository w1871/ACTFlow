import os
import torch
import torch.distributed as dist

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  #通信后端
    dist.init_process_group(backend=args.dist_backend,init_method=args.dist_url,world_size=args.world_size,rank=args.rank)
    # 等待gpu
    dist.barrier()

def cleanup():
    dist.destroy_process_group()

def is_main_process():
    return dist.get_rank()==0

def reduce_value(value,average=True):
    world_size = dist.get_world_size()
    if(world_size < 2):
        return value
    with torch.no_grad():
        dist.all_reduce(value)
        if(average):
            value /= world_size
        return value