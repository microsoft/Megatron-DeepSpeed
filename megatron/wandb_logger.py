from dataclasses import dataclass,field
from typing import Optional,List,Dict,Tuple, Union
import torch
from typing import Union
import wandb
from argparse import Namespace
from collections import defaultdict
import os
from warnings import warn

#TODO: do we want to use the watch? https://docs.wandb.ai/ref/python/watch
@dataclass
class WandBConfig(object):
    # compatibility to previous logger
    # config to be logged in wandb
    config:Optional[Namespace]=field(default_factory=lambda :None)
    # NOTE: these are ignored by wandb
    # filled from `tensorboard_log_interval`, frequency of logging
    log_interval:int=field(default=1)
    # filled from `tensorboard_queue_size`, ignored by wandb
    queue_size:int=field(default=int(1e3))
    # if set, log locally into this dir, gets filled from tensorboard_dir
    local_dir:Optional[str]=field(default=None)

    # wandb specific config
    # filled from args
    #filled with kwargs
    entity:str=field(default="meditron")
    #wandb project to log to
    project:str=field(default="meditron")
    # save the code to the notebook?
    save_code:bool=field(default=False)
    # stuff to filter by
    tags:Optional[List[str]]=field(default=None)
    # short descriptive name to quickly find things
    name:Optional[str]=field(default=None)
    # long form notes and info to store
    notes:Optional[str]=field(default=None)
    # TODO: discuss how we want to do this, do we want to resume logging?
    resume:str=field(default="allow")
    # TODO: can fill this from the environment variable `WANDB_RUN_ID` ?
    # globally unique id, if passed
    run_id:Optional[str]=field(default=None)
    # "magic" auto instrumentation, see https://docs.wandb.ai/ref/python/init
    magic:bool=field(default=False)
    api_key:Optional[str]=field(default=None)
    with_tensorboard:bool=field(default=True)
    try_catch_guard:bool=field(default=True)

    @staticmethod
    def default(project:str,run_id:str=run_id):
        return WandBConfig(project=project,run_id=run_id)

    @staticmethod
    def from_args(args)->'WandBConfig':
        assert args.rank==args.world_size-1, f"Only supposed to launch on rank {args.rank+1}, but got {args.rank}"
        # following the megatron setup for now, could also do groups instead: https://docs.wandb.ai/guides/track/log/distributed-training
        return WandBConfig(local_dir=args.tensorboard_dir,
                           queue_size=args.tensorboard_queue_size,
                           log_interval=args.log_interval,
                           config=args,entity=args.wandb_entity,
                           project=args.wandb_project,
                           run_id=args.wandb_id,
                           resume=args.wandb_resume,
                           api_key=args.wandb_api_key,
                           try_catch_guard=False,
                           with_tensorboard=True)

import functools
# dummy_named just because of bare *
def try_catch_guard(_func=None,*,dummy_named=None,**decorator_kwargs):
    def decorator_try_catch_guard(func):
        @functools.wraps(func)
        def try_catch_wrapper(*args,**kwargs):
            s=args[0]
            if s.cfg.try_catch_guard:
                try:
                    return func(*args,**kwargs)
                except BaseException as e:
                    warn(f"Ignoring error {e} in WandbTBShim")
            else:
                return func(*args,**kwargs)
        return try_catch_wrapper
    if _func is None:
        return decorator_try_catch_guard
    else:
        return decorator_try_catch_guard(_func)
    

class WandbTBShim(object):
    """
    Shim class that holds the configuration we want the wandb wrapper to use
    (e.g. to control sampling, delay upload etc) and that translates the API
    """
    def __init__(self, config:WandBConfig):
        super().__init__()
        self.cfg=config
        if os.environ.get("WANDB_API_KEY") is None:
            if self.cfg.api_key is None:
                raise ValueError("WANDB_API_KEY is not set, nor passed as an argument")
            else:
                os.environ["WANDB_API_KEY"]=self.cfg.api_key
        wandb.init(config=config.config,
                   entity=config.entity,
                   project=config.project,
                   save_code=config.save_code,
                   tags=config.tags,
                   name=config.name,
                   notes=config.notes,
                   resume=config.resume,
                   id=config.run_id,
                   dir=config.local_dir
                )
        self._last_step = None
        self._log_accum = {}
        if self.cfg.with_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                print('> setting tensorboard ...')
                self.tb_writer = SummaryWriter(
                    log_dir=config.local_dir,
                    max_queue=config.queue_size)
            except ModuleNotFoundError:
                print('WARNING: TensorBoard writing requested but is not '
                      'available (are you using PyTorch 1.1.0 or later?), '
                      'no TensorBoard logs will be written.', flush=True)
        else:
            self.tb_writer=None

    @try_catch_guard
    def _add_scalar(self, name: str, var: Union[float, int, torch.Tensor], step: int):
        if isinstance(var, torch.Tensor):
            var = var.item()
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(name, var, global_step=step)
        if " vs " in name:
            # wandb does not allow logging to previous steps and the ' vs '
            # scalars are usually a lot of steps forward compared to the rest
            # of the scalars (as they count per sample, not per batch) so we
            # just ignore them and rely on tensorboard to log them
            warn(f"Ignoring wandb log for {name}")
            return

        if self._last_step is not None and step > self._last_step:
            self.flush_all()

        self._last_step = step
        self._log_accum[name] = var

    @try_catch_guard
    def add_scalar(self, name: str, var: Union[float, int, torch.Tensor], step: int):
        if isinstance(var, torch.Tensor):
            var = var.item()
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(name, var, global_step=step)
        
        wandb.log({name:var},step=step)

    @try_catch_guard
    def flush_all(self):
        if len(self._log_accum) > 0:
            wandb.log(self._log_accum, step=self._last_step, commit=True)
        self._log_accum = {}
        self._last_step = None

    @try_catch_guard
    def add_text(self,name:str,value:str):
        # we log this on the creation of the wandb object, hence no need to log it here
        if self.tb_writer is not None:
            self.tb_writer.add_text(name,value)

def toy_test(writer):
    for i,l in zip(range(10),range(10,20)):
        r=10-i
        for k in range(5):
            writer.add_scalar(f"forward{k}",l,i)
            writer.add_scalar(f"backward{k}",r,i)
            writer.add_scalar(f"forward{k} vs forward",l,i)
            writer.add_scalar(f"forward{k} vs backward",i,r)
if __name__=="__main__":
    writer=WandbTBShim(WandBConfig.default("wandb-toy-test",run_id="meditron-wandb-test"))
    toy_test(writer)
