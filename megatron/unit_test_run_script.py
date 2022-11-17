import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from megatron.initialize import initialize_megatron
import importlib
from megatron import fused_kernels
import torch
from torch.testing._internal import common_utils
import unittest
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from itertools import product


global fused_mix_prec_layer_norm_cuda
fused_mix_prec_layer_norm_cuda = None

initialize_megatron(extra_args_provider=None,
                        args_defaults={})

fused_mix_prec_layer_norm_cuda = importlib.import_module(
          "fused_mix_prec_layer_norm_cuda")

from model.fused_layer_norm import MixedFusedLayerNorm

def _prep_inputs(batch_size, normalized_shape, dtype):
    shape = (batch_size, *normalized_shape)
    fused = torch.randn(shape).cuda().requires_grad_(True)
    with torch.no_grad():
        native = fused.clone().to(dtype).requires_grad_(True)
    return native, fused

autocast_dtypes = (torch.half, torch.bfloat16) if torch.cuda.is_bf16_supported() else (torch.half,)

class TestFusedLayerNorm(unittest.TestCase):

    def _test_fused_layer_norm(
        self, batch_size, contiguous, elementwise_affine, mixed_fused, dtype,
        fwd_thresholds=dict(rtol=None, atol=None), bwd_thresholds=dict(rtol=None, atol=None)
        ):

        normalized_shape = [32, 16]

        if not mixed_fused:
            assert True
            """
            module_cpu_ = FusedLayerNorm(
                normalized_shape=normalized_shape, elementwise_affine=elementwise_affine).cpu()
            module_cuda_ = FusedLayerNorm(
                normalized_shape=normalized_shape, elementwise_affine=elementwise_affine).to(device="cuda", dtype=dtype)
            """
        else:
            assert elementwise_affine
            module_cpu_ = MixedFusedLayerNorm(
                normalized_shape=normalized_shape).cpu()
            module_cuda_ = MixedFusedLayerNorm(
                normalized_shape=normalized_shape).to(device="cuda", dtype=dtype)

        torch.cuda.manual_seed(42)
        if contiguous:
            input_shape = [batch_size] + normalized_shape
            input_ = torch.randn(input_shape, device="cpu").requires_grad_(True)
            input_cuda_ = input_.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
            self.assertTrue(input_.is_contiguous())
            self.assertTrue(input_cuda_.is_contiguous())
        else:
            input_shape = [batch_size] + normalized_shape
            input_shape = [batch_size * 3] + [normalized_shape[0] * 5, normalized_shape[1] * 3]
            input_src_ = torch.randn(input_shape, device="cpu")
            input_ = input_src_[::3, ::5, ::3].detach().requires_grad_(True)
            input_cuda_ = input_src_.to(device="cuda", dtype=dtype)[::3, ::5, ::3].detach().requires_grad_(True)
            # make sure that tensors are NOT contiguous.
            self.assertFalse(input_.is_contiguous())
            self.assertFalse(input_cuda_.is_contiguous())
        out_cpu_ = module_cpu_(input_)
        gO = torch.rand_like(out_cpu_)
        out_cpu_.backward(gO)
        out_cuda_ = module_cuda_(input_cuda_)

        gO = gO.to(device="cuda", dtype=dtype)
        out_cuda_.backward(gO)
        self.assertFalse(out_cpu_.is_cuda)
        self.assertTrue(out_cuda_.is_cuda)
        torch.testing.assert_close(
            out_cpu_.to(device="cuda", dtype=dtype), out_cuda_, **fwd_thresholds)
        torch.testing.assert_close(
            input_.grad.to(device="cuda", dtype=dtype), input_cuda_.grad, **bwd_thresholds)
    
    def test_layer_norm_mixed_1(self, batch_size=16, contiguous=True, elementwise_affine=True, mixed_fused=True, dtype=torch.float):
        self._test_fused_layer_norm(batch_size, contiguous, elementwise_affine, mixed_fused, dtype)
    #def test_layer_norm_mixed_2(self, batch_size=65536, contiguous=False, elementwise_affine=True, mixed_fused=True, dtype=torch.float):
    #    self._test_fused_layer_norm(batch_size, contiguous, elementwise_affine, mixed_fused, dtype)


#instantiate_device_type_tests(TestFusedLayerNorm, globals(), only_for=("cuda",))

if __name__ == "__main__":
    print('cmd entry:', sys.argv)
    sys.argv = [sys.argv[0]]
    print('cmd entry:', sys.argv)
    import subprocess
    instantiate_device_type_tests(TestFusedLayerNorm, globals(), only_for=("cuda",))
    unittest.main()
