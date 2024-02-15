from light.zbpp_light.weight_grad_store import WeightGradStore

from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.utils.timer import BACKWARD_MICRO_TIMER, \
    BACKWARD_GLOBAL_TIMER, BACKWARD_INNER_MICRO_TIMER, BACKWARD_INNER_GLOBAL_TIMER
from deepspeed.runtime.utils import PartitionedTensor
from deepspeed.accelerator import get_accelerator

import torch
from torch.cuda.amp import custom_bwd
from packaging import version


from megatron.core.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_global_memory_buffer,
)


def gradientUpdateFunction(total_input, grad_output, weight):
    if weight.grad == None:
        weight.grad = grad_output.t().matmul(total_input)
    else:
        weight.grad += grad_output.t().matmul(total_input)

@staticmethod
@custom_bwd
def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    use_bias = ctx.use_bias

    if ctx.sequence_parallel:
        world_size = get_tensor_model_parallel_world_size()
        dim_size = list(input.size())
        dim_size[0] = dim_size[0] * world_size

        all_gather_buffer = \
            get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")

        if version.parse(torch.__version__) >= version.parse('1.13'):
            handle = torch.distributed.all_gather_into_tensor(
                all_gather_buffer,
                input,
                group=get_tensor_model_parallel_group(), async_op=True)
        else:
            handle = torch.distributed._all_gather_base(
                all_gather_buffer,
                input,
                group=get_tensor_model_parallel_group(), async_op=True)

        # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
        # gather is scheduled before the input gradient computation
        total_input = all_gather_buffer
    else:
        total_input = input
    grad_input = grad_output.matmul(weight)

    if ctx.sequence_parallel:
        handle.wait()

    # Doing gather + slicing during the NeMo forward pass can make this tensor
    # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
    # clones it if it's not contiguous:
    # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
    grad_output = grad_output.contiguous()
    # Convert the tensor shapes to 2D for execution compatibility
    if len(grad_output.shape) == 3:
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1],
                                    grad_output.shape[2])
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1],
                    total_input.shape[2])
    else:
        # Somehow when DeepSpeed MoE is used, grad_output could have 4 dimensions.
        # TODO: May need further investigation
        total_input = total_input.contiguous()
        grad_output = grad_output.view(-1, grad_output.shape[-1])
        total_input = total_input.view(-1, total_input.shape[-1])

    if ctx.async_grad_allreduce:
        # Asynchronous all-reduce
        handle = torch.distributed.all_reduce(
                grad_input, group=get_tensor_model_parallel_group(), async_op=True)
        # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
        # all-reduce is scheduled before the weight gradient computation

    if ctx.sequence_parallel:
        assert not ctx.async_grad_allreduce
        dim_size = list(input.size())
        sub_grad_input = torch.empty(dim_size, dtype=input.dtype,
                                        device=get_accelerator().current_device_name(),
                                        requires_grad=False)
        # reduce_scatter
        handle = torch.distributed._reduce_scatter_base(sub_grad_input, grad_input,
                                                        group=get_tensor_model_parallel_group(),
                                                        async_op=True)
        # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
        # reduce scatter is scheduled before the weight gradient computation

    # TODO: temporary commented
    # if ctx.gradient_accumulation_fusion:
    #     if weight.main_grad.dtype == torch.float32:
    #         fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, weight.main_grad)
    #     elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
    #         fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input, grad_output, weight.main_grad)
    #     else:
    #         raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
    #     grad_weight = None
    # else:
    #     grad_weight = grad_output.t().matmul(total_input)
    # grad_weight = grad_output.t().matmul(total_input)

    from light.zbpp_light.weight_grad_store import WeightGradStore
    WeightGradStore.put(total_input, grad_output, weight, gradientUpdateFunction)
    grad_weight = None
    
    grad_bias = grad_output.sum(dim=0) if use_bias else None

    if ctx.sequence_parallel:
        handle.wait()
        return sub_grad_input, grad_weight, grad_bias, None, None, None

    if ctx.async_grad_allreduce:
        handle.wait()

    return grad_input, grad_weight, grad_bias, None, None, None


def _exec_backward_only_pass(self, buffer_id):
    assert self.optimizer is not None, "must provide optimizer during " \
                                        "init in order to use backward"

    self.mem_status('BEFORE BWD ONLY', reset_max=True)
    from megatron.core.tensor_parallel.layers import LinearWithGradAccumulationAndAsyncCommunication
    original_backward = LinearWithGradAccumulationAndAsyncCommunication.backward
    LinearWithGradAccumulationAndAsyncCommunication.backward = backward
    # The last stage just runs backward on the loss using DeepSpeed's typical
    # mechanisms.
    if self.is_last_stage():
        super(PipelineEngine, self).backward(self.loss)
        WeightGradStore.flush()
        self.mem_status('AFTER BWD ONLY')

        LinearWithGradAccumulationAndAsyncCommunication.backward = original_backward
        return

    outputs = self.pipe_buffers['outputs'][buffer_id]

    if self.wall_clock_breakdown():
        self.timers(BACKWARD_MICRO_TIMER).start()
        self.timers(BACKWARD_GLOBAL_TIMER).start()
        self.timers(BACKWARD_INNER_MICRO_TIMER).start()
        self.timers(BACKWARD_INNER_GLOBAL_TIMER).start()

    # Reconstruct if we previously partitioned the output. We must be
    # careful to also restore the computational graph of the tensors we partitioned.
    if self.is_pipe_partitioned:
        if self.is_grad_partitioned:
            if self.pipe_partition_output_meta_cache is None:
                self.pipe_partition_output_meta_cache = outputs[0].to('cpu')
            part_output = PartitionedTensor.from_meta(meta=self.pipe_partition_output_meta_cache,
                                                        local_part=outputs[1],
                                                        group=self.grid.get_slice_parallel_group())
            self.pipe_buffers['output_tensors'][buffer_id].data = part_output.full()
            outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[2:])
        else:
            # Already restored from partition
            self.pipe_buffers['output_tensors'][buffer_id].data = outputs[0]
            outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[1:])

    grad_tensors = self.grad_layer
    if self.is_grad_partitioned:
        #print(f'RANK={self.global_rank} BEFORE-BWD restoring grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')
        if self.grad_partition_grad_layer_meta_cache is None:
            self.grad_partition_grad_layer_meta_cache = self.grad_layer[0].to('cpu')
        part_grad = PartitionedTensor.from_meta(meta=self.grad_partition_grad_layer_meta_cache,
                                                local_part=self.grad_layer[1],
                                                group=self.grid.get_slice_parallel_group())
        grad_tensors = (part_grad.full(), *grad_tensors[2:])
        part_grad = None
        #print(f'RANK={self.global_rank} BEFORE-BWD restored grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')

    if self.using_bf16_optimizer and not self.is_last_stage():
        # manually call because we don't call optimizer.backward()
        self.optimizer.clear_lp_grads()

    # This handles either a single tensor or tuple of tensors.
    
    if isinstance(outputs, tuple):
        out_tensors = [t for t in outputs if t.is_floating_point()]
        assert len(out_tensors) == len(grad_tensors)
        torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
    else:
        torch.autograd.backward(tensors=(outputs, ), grad_tensors=(grad_tensors, ))
    

    WeightGradStore.flush()

    if self.using_bf16_optimizer and not self.is_last_stage():
        # manually call because we don't call optimizer.backward()
        self.optimizer.update_hp_grads(clear_lp_grads=False)

    # Free up the memory from the output of forward()
    self.pipe_buffers['output_tensors'][buffer_id] = None
    self.pipe_buffers['outputs'][buffer_id] = None
    grad_tensors = None
    
    LinearWithGradAccumulationAndAsyncCommunication.backward = original_backward

    if self.wall_clock_breakdown():
        self.timers(BACKWARD_INNER_MICRO_TIMER).stop()
        self.timers(BACKWARD_INNER_GLOBAL_TIMER).stop()
        self.timers(BACKWARD_MICRO_TIMER).stop()
        self.timers(BACKWARD_GLOBAL_TIMER).stop()

def _exec_weight_pass(self):
    if self.using_bf16_optimizer:
        # manually call because we don't call optimizer.backward()
        self.optimizer.clear_lp_grads()
    WeightGradStore.pop()
    if self.using_bf16_optimizer:
        self.optimizer.update_hp_grads(clear_lp_grads=False)