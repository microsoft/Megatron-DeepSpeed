import math
import numbers

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init


class LayerNorm1P(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, apply_layernorm_1p=False):
        super(LayerNorm1P, self).__init__()
        self.eps = eps
        self.apply_layernorm_1p = apply_layernorm_1p
        
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()
    
    def reset_parameters(self):

        if self.apply_layernorm_1p:
            init.zeros_(self.weight)
            init.zeros_(self.bias)
        else:
            init.ones_(self.weight)
            init.zeros_(self.bias)
    
    def forward(self, input):
        if self.apply_layernorm_1p:
            weight_plus_1 = (self.weight + 1)
            output = torch.nn.functional.layer_norm(input, self.normalized_shape, weight_plus_1, self.bias, self.eps)
            return output
        else:
            return torch.nn.functional.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
