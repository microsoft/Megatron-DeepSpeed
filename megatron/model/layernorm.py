import numbers

import torch
from torch.nn.parameter import Parameter

class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, sequence_parallel=False):
        super(LayerNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.ones(normalized_shape))
        self.bias = Parameter(torch.zeros(normalized_shape))
        self.sequence_parallel = sequence_parallel
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)

    def forward(self, x):
        output = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return output
