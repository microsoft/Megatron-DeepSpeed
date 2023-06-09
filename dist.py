"""
dist.py

Contains methods for initializing distributed communication.
"""
from __future__ import absolute_import, annotations, division, print_function

import os

from typing import Optional, Callable
from mpi4py import MPI

import logging

log = logging.getLogger(__name__)


BACKENDS = [
    'deepspeed',
    'ds',
    'ddp',
    'horovod',
    'hvd',
]


def seed_everything(seed: int):
    import torch
    import numpy as np
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_deepspeed():
    import deepspeed
    deepspeed.init_distributed()


def init_process_group(
        rank: int | str,
        world_size: int | str,
        backend: Optional[str] = None,
) -> None:
    import torch
    import torch.distributed as dist
    if torch.cuda.is_available():
        backend = 'nccl' if backend is None else str(backend)
    else:
        backend = 'gloo' if backend is None else str(backend)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=int(rank),
            world_size=int(world_size),
            init_method='env://',
        )


def run_ddp(fn: Callable, world_size: int) -> None:
    import torch.multiprocessing as mp
    mp.spawn(  # type:ignore
        fn,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


def get_rank() -> int:
    return int(MPI.COMM_WORLD.Get_rank())


def get_world_size() -> int:
    return int(MPI.COMM_WORLD.Get_size())


def get_local_rank() -> int:
    return int(os.environ.get(
        'PMI_LOCAL_RANK',  # PMI_* for Polaris (/ Aurora) @ ALCF
        os.environ.get(    # OMPI_* for ThetaGPU @ ALCF
            'OMPI_COMM_WORLD_LOCAL_RANK',
            os.environ.get(
                'LOCAL_RANK',
                '0'
            )
        )
    ))


def query_environment() -> dict[str, int]:
    """Query environment variables for info about distributed setup"""
    ws = os.environ.get('WORLD_SIZE', None)
    r = os.environ.get('RANK', None)
    lr = os.environ.get('LOCAL_RANK', None)
    if ws is not None and r is not None and lr is not None:
        return {
            'world_size': int(ws),
            'rank': int(r),
            'local_rank': int(lr)
        }

    return {
        'world_size': int(get_world_size()),
        'rank': int(get_rank()),
        'local_rank': int(get_local_rank()),
    }


def setup_torch_DDP(port: str = '2345') -> dict[str, int]:
    import torch
    rank = os.environ.get('RANK', None)
    world_size = os.environ.get('WORLD_SIZE', None)
    local_rank = os.environ.get('LOCAL_RANK', None)

    import socket
    world_size = int(get_world_size())
    rank = int(get_rank())
    local_rank = int(get_local_rank())
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    master_addr = (
        socket.gethostname() if rank == 0 else None
    )
    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    os.environ['MASTER_ADDR'] = master_addr
    if (eport := os.environ.get('MASTER_PORT', None)) is None:
        os.environ['MASTER_PORT'] = port
    else:
        os.environ['MASTER_PORT'] = eport
        if rank == 0:
            log.info(f'Caught MASTER_PORT:{eport} from environment!')
    init_process_group(
        rank=rank,
        world_size=world_size,
        backend='nccl' if torch.cuda.is_available() else 'gloo'
    )
    return {'world_size': world_size, 'rank': rank, 'local_rank': local_rank}


def setup_torch_distributed(
        backend: str,
        port: str = '2345',
) -> dict:
    import torch
    rank = os.environ.get('RANK', None)
    world_size = os.environ.get('WORLD_SIZE', None)
    # local_rank = os.environ.get(
    #     'PMI_LOCAL_RANK',
    #     os.environ.get(
    #         'OMPI_COMM_WORLD_LOCAL_RANK',
    #         None
    #     )
    # )
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    be = backend.lower()
    assert be in BACKENDS
    if rank == 0 and local_rank == 0:
        log.info(f'Using {backend} for distributed training')
    if be in ['ddp', 'DDP']:
        dsetup = setup_torch_DDP(port)
        world_size = dsetup['world_size']
        rank = dsetup['rank']
        local_rank = dsetup['local_rank']
    elif be in ['deepspeed', 'ds']:
        init_deepspeed()
        world_size = get_world_size()
        rank = get_rank()
        local_rank = get_local_rank()
    elif be in ['horovod', 'hvd']:
        import horovod.torch as hvd
        hvd.init() if not hvd.is_initialized() else None
        rank = hvd.rank()
        world_size = hvd.size()
        local_rank = hvd.local_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(hvd.local_rank())
    else:
        raise ValueError
    os.environ['world_size'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    return {'world_size': world_size, 'rank': rank, 'local_rank': local_rank}


def setup_torch(
        backend: str = 'deepspeed',
        port: str = '2345',
        seed: Optional[int] = None,
) -> int:
    import torch
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True     # type:ignore
    torch.backends.cudnn.benchmark = True         # type:ignore
    torch.backends.cudnn.allow_tf32 = True        # type:ignore
    torch.backends.cuda.matmul.allow_tf32 = True  # type:ignore
    torch.use_deterministic_algorithms(True)
    dsetup = setup_torch_distributed(backend=backend, port=port)
    rank = dsetup['rank']
    world_size = dsetup['world_size']
    local_rank = dsetup['local_rank']
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    nthreads = os.environ.get('OMP_NUM_THREADS', None)
    if nthreads is not None:
        torch.set_num_threads(int(nthreads))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    log.info(f'Global Rank: {rank} / {world_size-1}')
    if seed is not None:
        seed_everything(seed * (rank + 1) * (local_rank + 1))
    return rank


def cleanup() -> None:
    import torch.distributed as tdist
    tdist.destroy_process_group()
