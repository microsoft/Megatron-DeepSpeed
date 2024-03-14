# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Utilities for models."""

import math

import torch

from megatron import get_args

from deepspeed.runtime.zero import GatheredParameters

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_

# Implement Sliding Window
def modify_attention_mask_for_sliding_window(attention_mask, window_size):
    """
    Modifies the attention mask for sliding window attention.

    Args:
        attention_mask (torch.Tensor): Original attention mask of shape [batch_size, seq_len].
        window_size (int): The size of the sliding window.

    Returns:
        torch.Tensor: Modified attention mask.
    """
    batch_size, seq_len = attention_mask.shape
    new_mask = torch.full((batch_size, seq_len, seq_len), float('-inf'), device=attention_mask.device)

    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        new_mask[:, i, start:end] = attention_mask[:, start:end]

    return new_mask


def gather_and_init(param, init_method):
    with GatheredParameters(param, modifier_rank=0):
        init_method(param)


def attention_mask_func(attention_scores, attention_mask):
    args = get_args()
    if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
        attention_mask_ = attention_mask
        actual_seqlen = attention_scores.size()[2]
        if actual_seqlen != attention_mask_.size()[2]:
            # attention_mask has size [1, 1, seqlen, seqlen]
            attention_mask_ = attention_mask_[:, :, :actual_seqlen, :actual_seqlen].contiguous()
        attention_scores.masked_fill_(attention_mask_, -10000.0)
    else:
        attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def get_linear_layer(rows, columns, init_method, gather_params_on_init=False):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if get_args().perform_initialization:
        with GatheredParameters(layer.weight, modifier_rank=0, enabled=gather_params_on_init):
            init_method(layer.weight)
    with torch.no_grad():
        with GatheredParameters(layer.bias, modifier_rank=0, enabled=gather_params_on_init):
            layer.bias.zero_()
    return layer

@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))
def openai_gelu(x):
    return gelu_impl(x)

#This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype)+torch.ones_like(x).to(dtype=x.dtype))
