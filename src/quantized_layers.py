# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
## Inspired by https://github.com/eladhoffer/quantized.pytorch/blob/master/models/modules/quantize.py


import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter
from torch import Tensor
from torch.nn.utils import parametrize
from torch.nn.modules import Module
import math

from utils import make_pow2_hadamard_matrix

from deel.torchlip import SpectralLinear #, SpectralLinearNoBjorck


class UniformQuantize(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None,
                #stochastic=False, inplace=False, enforce_true_quantized=True, enforce_true_zero=False, num_chunks=None, out_half=False
        ):
        output = input.clone()
        coefInt = float(2**(num_bits-1))
        alpha = max(abs(min_value),abs(max_value))
        if alpha == 0.:
            alpha = 1.
        output.mul_(coefInt/alpha).round_() #quantize
        if num_bits == 1:   #ternary
            output.clamp_(-coefInt,coefInt)
        else:
            output.clamp_(-coefInt,coefInt-1) #quantize
        output.mul_(alpha/coefInt)  #dequantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None


def quantize(x, num_bits=8, min_value=None, max_value=None):
    #, num_chunks=None, stochastic=False, inplace=False, enforce_true_quantized = True):
    return UniformQuantize().apply(x, num_bits, min_value, max_value)


class QLinear(nn.Linear):
    """ Layer that uniformly quantize the weight at forward time over 2^num_bits values in the range
        [-max(|weight|), max(|weight|)]. It allows full-precision weights in the case num_bits = None.
    """
    
    def __init__(self, in_features, out_features, bias=True, num_bits=8, quantized_minus_id=False):
        super(QLinear, self).__init__(in_features, out_features, bias) 
        
        if self.weight.shape[0] == self.weight.shape[1]:
            torch.nn.init.orthogonal_(self.weight)
        else:
            nn.init.xavier_uniform_(self.weight)
                  
        self.num_bits = num_bits
        #self.enforce_true_quantized = True
        self.quantized_minus_id = quantized_minus_id
        
        if self.num_bits is not None:
            if self.quantized_minus_id:
                self.delta_m = torch.eye(self.weight.shape[0], self.weight.shape[1])
            else:
                self.delta_m = torch.zeros(self.weight.shape[0], self.weight.shape[1])
            quantize_register(self,
                    name = 'weight',
                    num_bits = num_bits,
                    #enforce_true_quantized = self.enforce_true_quantized,
                    delta_m = self.delta_m,
                    )


def quantize_register(module: Module,
                  name: str = 'weight',
                  num_bits: int = 8,
                  #enforce_true_quantized: bool = True,
                  delta_m: torch.Tensor = None,
                  ) -> Module:
    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(
            f"Module '{module}' has no parameter or buffer with name '{name}'"
        )

    parametrize.register_parametrization(module, name, _Quantize(
        weight, 
        num_bits=num_bits, 
        #enforce_true_quantized=enforce_true_quantized,
        delta_m=delta_m,))
    return module

class _Quantize(Module):
    def __init__(
        self,
        weight: torch.Tensor,
        num_bits: int = 8, 
        #enforce_true_quantized: bool = True,
        delta_m: torch.Tensor = None,
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        #self.enforce_true_quantized = enforce_true_quantized
        self.delta_m = delta_m


    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        self.delta_m = self.delta_m.to(weight.device)
        qweight = quantize(weight-self.delta_m, num_bits=self.num_bits,
                                min_value=float(weight.min()),
                                max_value=float(weight.max()),
                                #enforce_true_quantized=self.enforce_true_quantized
                                ) \
                       + self.delta_m
        return qweight

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        # we may want to assert here that the passed value already
        # satisfies constraints
        return value



class QSpectralLinear(SpectralLinear):
    def __init__(self, in_features, out_features, bias=True, num_bits=8, quantized_minus_id = False):
        super(QSpectralLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.qweight = self.weight.clone()
        #self.enforce_true_quantized = True
        self.quantized_minus_id = quantized_minus_id
        if self.quantized_minus_id:
            self.delta_m = torch.eye(self.weight.shape[0], self.weight.shape[1])
        else:
            self.delta_m = torch.zeros(self.weight.shape[0], self.weight.shape[1])
        quantize_register(self,
                  name = 'weight',
                  num_bits = num_bits,
                  #enforce_true_quantized = self.enforce_true_quantized,
                  delta_m = self.delta_m,
                  )



class QSpectralLinearNoBjorck(SpectralLinear):
    def __init__(self, in_features, out_features, bias=True, num_bits=8, quantized_minus_id = False):
        super(QSpectralLinearNoBjorck, self).__init__(in_features, out_features, bias, eps_bjorck = None)
        self.num_bits = num_bits
        self.qweight = self.weight.clone()
        #self.enforce_true_quantized = True
        self.quantized_minus_id = quantized_minus_id
        if self.quantized_minus_id:
            self.delta_m = torch.eye(self.weight.shape[0], self.weight.shape[1])
        else:
            self.delta_m = torch.zeros(self.weight.shape[0], self.weight.shape[1])
        quantize_register(self,
                  name = 'weight',
                  num_bits = num_bits,
                  #enforce_true_quantized = self.enforce_true_quantized,
                  delta_m = self.delta_m,
                  )



class Binadamard(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32, basic_block=None, normalize=True):
        #torch.manual_seed(88)
        super(Binadamard, self).__init__(in_features, out_features, bias)
        self.units = in_features
        self.quantized = True
        if basic_block == 'full':
            self.basic_block = make_pow2_hadamard_matrix(torch.log2(torch.tensor(self.units)), normalize=normalize)
        else:
            self.basic_block = make_pow2_hadamard_matrix(basic_block, normalize=normalize)
        if normalize: self.scale = 1.
        else: self.scale = 1. / torch.sqrt(torch.tensor(2**basic_block, dtype=torch.float32))
        self.dtype = dtype
        self.num_bits = 2
        if self.basic_block is not None:
            self.block_hadamard_kernel = self.make_block_hadamard_basis()

        self.sign_vector = Parameter(
            torch.empty(1, self.units), requires_grad=True
        )
        nn.init.xavier_uniform_(self.sign_vector) ####,gain=0.01)

        parametrize.register_parametrization(self, 'weight', binadamard_parametrization(
            self.sign_vector,
            self.block_hadamard_kernel,
            device = self.weight.device)
        )

    def make_block_hadamard_basis(self):
        if self.basic_block is None: return None
        m = self.basic_block.shape[0]
        block_hadamard_kernel = torch.zeros((self.units, self.units))
        assert self.units % m == 0, (
            "number of units (kernel size n) should be a multiple of the basic block "
            "size m but found n = {} and m = {}".format(self.units, m))
        rep = int(self.units / m)
        block_hadamard_kernel = torch.block_diag(*(rep * [self.basic_block]))

        return block_hadamard_kernel

    def forward(self, input):
        if input is None:
            if self.bias is not None:
                return self.bias
            return torch.zeros(self.units, device=self.weight.device)
        return self.scale * F.linear(input, self.weight, self.bias)

    def apply_weight_constraints_(self):
	    with torch.no_grad():  # Ensure no gradient computation is being recorded
	        self.sign_vector.data.clamp_(-1, 1)  # In-place clamping of weights


class BinaryNorm(nn.Linear):

    def __init__(self, in_features, out_features, bias=False, dtype=torch.float32):
        super(BinaryNorm, self).__init__(in_features, out_features, bias)
        self.units = in_features
        self.quantized = True
        self.dtype = dtype
        self.num_bits = 2
        self.norm = 1 / torch.sqrt(torch.tensor(self.units))
        self.original_weight = Parameter(
            torch.empty(self.units, self.units), requires_grad=True
        )
        nn.init.xavier_uniform_(self.original_weight)
        parametrize.register_parametrization(
            self, 'weight', binary_parametrization(
                self.original_weight,
                self.norm,
                self.original_weight.device)
        )

    def forward(self, input):
        qinput = input
        qbias = self.bias

        #self.qweight = self.norm * BinaryQuantize().apply(self.weight)

        #output = self.norm * F.linear(qinput, self.weight, qbias)
        output = F.linear(qinput, self.weight, qbias)
        return output



class binadamard_parametrization(Module):
    def __init__(
        self,
        sign_vector: torch.Tensor,
        block_hadamard_kernel: torch.Tensor,
        device = None,
    ) -> None:
        super().__init__()
        self.sign_vector = sign_vector
        self.device = device
        self.block_hadamard_kernel = block_hadamard_kernel

    def make_hadamard_kernel(self):
        if self.block_hadamard_kernel.device != self.device:
            self.block_hadamard_kernel = self.block_hadamard_kernel.to(self.device)
        H = self.block_hadamard_kernel
        D = self.make_sign_kernel(BinaryQuantize().apply(self.sign_vector))
        W = torch.matmul(D, H)
        return W

    def make_sign_kernel(self, vector):
        K = torch.diag(vector.view(-1))
        return K.requires_grad_(True)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.device != weight.device:
            self.sign_vector = self.sign_vector.to(weight.device)
            self.block_hadamard_kernel = self.block_hadamard_kernel.to(weight.device)
            self.device = weight.device

        qweight = self.make_hadamard_kernel()
        return qweight

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        # we may want to assert here that the passed value already
        # satisfies constraints
        return value





class BinaryQuantize(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, stochastic=False, inplace=False):
        ctx.inplace = inplace
        ctx.stochastic = stochastic

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        output = torch.sign(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input



