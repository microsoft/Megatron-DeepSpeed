# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from deepspeed.accelerator import get_accelerator
import torch

from typing import Callable, Any, Iterable, Union
from megatron import get_args

from .distrib_optimizer import DistributedOptimizer
from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer


import ezpz as ez
RANK = ez.get_rank()


def get_param_groups(
        modules: Union[torch.nn.Module, Iterable[torch.nn.Module]],
        no_weight_decay_cond: Callable[[str, torch.Tensor], bool],
        scale_lr_cond: Callable[[str, torch.Tensor], bool],
        lr_mult: Any,
        use_galore: bool = False,
):
    """
    Creates param groups (regularized vs non) based on:

    - weight decay condition.
    - learning rate scale condition (args.lr vs lr_mult * args.lr)
    - scale_lr_cond is used during finetuning, where head of the network
      requires a scaled version of the base learning rate.
    # if 'galore' in args.optimizer.lower():
    #     # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
    #     galore_params = []
    #     target_modules_list = ["attn", "mlp"]
    #     # for module_name, module in param_groups:
    #     for group_id, group in enumerate(param_groups):
    #         for param, p in enumerate(group['params']):
    #             if not isinstance(module, torch.nn.Linear):
    #                 continue
    #             if not any(target_key in module_name for target_key in target_modules_list):
    #                 continue
    #             print('enable GaLore for weights in module: ', module_name)
    #             galore_params.append(module.weight)
    #     id_galore_params = [id(p) for p in galore_params]
    #     # make parameters without "rank" to another group
    #     regular_params = [p for p in param_groups if id(p) not in id_galore_params]
    #     # then call galore_adamw
    #     param_groups = [
    #         {
    #             'params': regular_params
    #         },
    #         {
    #             'params': galore_params,
    #             'rank': RANK,
    #             'update_proj_gap': args.update_proj_gap,
    #             'scale': args.galore_scale,
    #             'proj_type': args.proj_type
    #         }
    #     ]
    """
    wd_no_scale_lr = []
    wd_scale_lr = []
    no_wd_no_scale_lr = []
    no_wd_scale_lr = []
    galore_params = []
    target_modules_list = ["attn", "mlp"]
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            if no_weight_decay_cond is not None:
                no_wd = no_weight_decay_cond(name, param)
            else:
                # do not regularize biases nor Norm parameters
                no_wd = name.endswith(".bias") or len(param.shape) == 1

            if scale_lr_cond is not None:
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False

            if not no_wd and not scale_lr:
                wd_no_scale_lr.append(param)
            elif not no_wd and scale_lr:
                wd_scale_lr.append(param)
            elif no_wd and not scale_lr:
                no_wd_no_scale_lr.append(param)
            else:
                no_wd_scale_lr.append(param)

    param_groups = []
    if len(wd_no_scale_lr):
        param_groups.append({'name': 'wd_no_scale_lr', 'params': wd_no_scale_lr, 'wd_mult': 1.0, 'lr_mult': 1.0})
    if len(wd_scale_lr):
        param_groups.append({'name': 'wd_scale_lr', 'params': wd_scale_lr, 'wd_mult': 1.0, 'lr_mult': lr_mult})
    if len(no_wd_no_scale_lr):
        param_groups.append({'name': 'no_wd_no_scale_lr', 'params': no_wd_no_scale_lr, 'wd_mult': 0.0, 'lr_mult': 1.0})
    if len(no_wd_scale_lr):
        param_groups.append({'name': 'no_wd_scale_lr', 'params': no_wd_scale_lr, 'wd_mult': 0.0, 'lr_mult': lr_mult})

    return param_groups


def get_megatron_optimizer(
        model,
        no_weight_decay_cond=None,
        scale_lr_cond=None,
        lr_mult=1.0
):
    args = get_args()
    assert args is not None

    # Base optimizer.
    param_groups = get_param_groups(
        model,
        no_weight_decay_cond,
        scale_lr_cond,
        lr_mult
    )
    if args.create_moe_param_group:
        from deepspeed.moe.utils import (
            split_params_into_different_moe_groups_for_optimizer
        )
        param_groups = split_params_into_different_moe_groups_for_optimizer(
            param_groups
        )

    if args.cpu_optimizer:
        assert args.optimizer == 'adam', 'CPU offloading is for Adam'
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.AdamW
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )

    elif args.optimizer.lower() == "galore_adamw":
        from galore_torch import GaLoreAdamW, GaLoreAdamW8bit
        # redefine way to call galore_adamw
        optimizer = GaLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_adamw":
        # redefine way to call galore_adamw
        optimizer = GaLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    # implement adafactor
    elif args.optimizer.lower() == "adafactor":
        import transformers
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = transformers.optimization.Adafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # low-rank adafactor
    elif args.optimizer.lower() == "galore_adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # 8-bit Adam
    elif args.optimizer.lower() == "adam8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_adamw8bit":
        optimizer = GaLoreAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'galore_adamw8bit_per_layer':
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
        optimizer_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = GaLoreAdamW8bit([{'params': [p], 'rank': args.rank, 'update_proj_gap': args.update_proj_gap * 2, 'scale': args.galore_scale, 'proj_type': args.proj_type}], lr=args.lr, weight_decay=args.weight_decay)
                else:
                    optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=args.lr, weight_decay=args.weight_decay)
        # get scheduler dict
        scheduler_dict = {}
        from galore_torch.peft_pretraining import training_utils
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = training_utils.get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if p.grad is None: 
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()
        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
        layer_wise_flag = True
    elif str(args.optimizer) == 'ipex.lamb':
        from intel_extension_for_pytorch.optim._lamb import Lamb
        optimizer = Lamb(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    elif str(args.optimizer) == 'ipex.fusedlamb':
        from intel_extension_for_pytorch.optim._lamb import Lamb
        optimizer = Lamb(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            fused=True,
        )
    elif str(args.optimizer).lower() == 'ds.fusedlamb':
        from deepspeed.ops.lamb import FusedLamb
        optimizer = FusedLamb(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    elif str(args.optimizer).lower() == 'adamwschedulefree':
        import schedulefree
        optimizer = schedulefree.AdamWScheduleFree(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            warmup_steps=args.lr_warmup_iters,
            foreach=args.schedulefree_for_each,
        )
    elif str(args.optimizer).lower() == 'sgdschedulefree':
        import schedulefree
        optimizer = schedulefree.SGDScheduleFree(
            param_groups,
            lr=args.lr,
            momentum=args.sgd_momentum,
            weight_decay=args.weight_decay,
            warmup_steps=args.lr_warmup_iters,
            foreach=args.schedulefree_for_each,
        )
    elif str(args.optimizer).lower() == 'apex.adam':
        assert get_accelerator().device_name() == 'cuda'
        from apex.optimizers import FusedAdam as Adam
        optimizer = Adam(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps
        )
    elif str(args.optimizer).lower() == 'apex.sgd':
        from apex.optimizers import FusedSGD as SGD
        optimizer = SGD(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.sgd_momentum
        )
    elif str(args.optimizer).lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps
        )
    elif args.optimizer == 'adam':
        if args.ds_fused_adam:
            # global Adam
            from deepspeed.ops.adam import FusedAdam
            Adam = FusedAdam
        else:
            Adam = torch.optim.Adam
        optimizer = Adam(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps
        )
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.sgd_momentum
        )
    else:
        raise TypeError(f'{args.optimizer} optimizer is not supported.')
    if args.deepspeed:
        return optimizer
    # Determine whether the params have main-grad field.
    params_have_main_grad = False
    if args.use_contiguous_buffers_in_local_ddp:
        params_have_main_grad = True
    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    if args.fp16 or args.bf16 or args.use_distributed_optimizer:
        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None
        # Constant loss scale.
        if args.loss_scale:
            grad_scaler = ConstantGradScaler(args.loss_scale)
        # Dynamic loss scale.
        else:
            if args.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=args.initial_loss_scale,
                    min_scale=args.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=args.loss_scale_window,
                    hysteresis=args.hysteresis)
        # Megatron optimizer.
        opt_ty = (
                DistributedOptimizer if args.use_distributed_optimizer 
                else Float16OptimizerWithFloat16Params
        )
        return opt_ty(optimizer,
                      args.clip_grad,
                      args.log_num_zeros_in_grad,
                      params_have_main_grad,
                      args.use_contiguous_buffers_in_local_ddp,
                      args.fp16,
                      args.bf16,
                      args.params_dtype,
                      grad_scaler,
                      model)
    # FP32.
    return FP32Optimizer(
        optimizer,
        args.clip_grad,
        args.log_num_zeros_in_grad,
        params_have_main_grad,
        args.use_contiguous_buffers_in_local_ddp,
        model
    )
