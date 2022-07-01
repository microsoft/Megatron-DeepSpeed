# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from .initialize import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from .utils import split_tensor_along_last_dim


def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size()==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_

def _reduce_scatter(input_, dim):
    """Reduce scatter the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size()==1:
        return input_

    # All-reduce.
    input_ = input_.contiguous()
    total_chunks = get_tensor_model_parallel_world_size()
    this_chunk = get_tensor_model_parallel_rank()

    assert input_.shape[dim] % total_chunks == 0

    chunk_size = input_.shape[dim]// total_chunks

    input_list = [torch.narrow(input_, dim, i*chunk_size, chunk_size).contiguous() for i in range(total_chunks)]

    output = torch.zeros_like(input_list[this_chunk], memory_format=torch.contiguous_format)
    torch.distributed.reduce_scatter(output, input_list, group=get_tensor_model_parallel_group())
    
    return output

def _gather_first_dim(input_, dim=0):
    """Gather tensors and concatinate along the first dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_
    input_ = input_.contiguous()
    # Size and dimension.
    rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim).contiguous()
  
    return output

def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


def _drop_tokens(input_, dim=0):
    if get_tensor_model_parallel_world_size() == 1:
        return input_ 
    total_chunks = get_tensor_model_parallel_world_size()
    this_chunk = get_tensor_model_parallel_rank()
    assert input_.shape[dim] % total_chunks == 0
    chunk_size = input_.shape[dim]// total_chunks

    return torch.narrow(input_, dim, this_chunk*chunk_size, chunk_size)

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_
    
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)

class _ReduceScatterFromModelParallelRegion(torch.autograd.Function):
    """Reduce scatter output of self attention for MoE"""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_first_dim(grad_output)

class _AllGatherFromModelParallelRegion(torch.autograd.Function):
    """Reduce scatter output of self attention for MoE"""

    @staticmethod
    def symbolic(graph, input_, dim):
        return _gather_first_dim(input_, dim)
    
    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        return _gather_first_dim(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _drop_tokens(grad_output, ctx.dim), None

class _DropTokens(torch.autograd.Function):
    "Drop tokens (this is a hacky approach until we can do reduce scatter)"

    @staticmethod
    def symbolic(graph, input_, dim):
        return _drop_tokens(input_, dim)

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        return _drop_tokens(input_, dim)

    @staticmethod
    def backward(ctx, input_):
        return _gather_first_dim(input_, ctx.dim), None


# -----------------
# Helper functions.
# -----------------

def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)

def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)

def reduce_scatter_from_tensor_model_parallel_region(input_):
    return _ReduceScatterFromModelParallelRegion.apply(input_)

def all_gather_from_tensor_model_parallel_region(input_, dim=0):
    return _AllGatherFromModelParallelRegion.apply(input_, dim)

def drop_tokens(input_, dim=0):
    return _DropTokens.apply(input_, dim)

    