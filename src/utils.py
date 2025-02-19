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
"""
File: utils.py
Created Date: Tue Mar 15 2022
Author: Randall Balestriero and Bobak Kiani
-----
Last Modified: Tue Mar 15 2022 7:30:04 PM
Modified By: Randall Balestriero
-----
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import torch
import numpy as np
import os

from torch.nn.functional import cross_entropy
from torchmetrics import Metric



def is_quantized(kernel):
    return hasattr(kernel, "quantized")


def to_float32_torch_tensor(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return torch.tensor(result, dtype=torch.float32)

    return wrapper


@to_float32_torch_tensor
def make_pow2_hadamard_matrix(k, normalize=False):
    """make a 2**k hadamard matrix"""

    n = 2**k
    if n == 1:
        return np.array([[1]])
    else:
        H = make_pow2_hadamard_matrix(k-1, normalize)
        if normalize: return (1./np.sqrt(2)) * np.kron(H, np.array([[1,1],[1,-1]]))
        return np.kron(H, np.array([[1,1],[1,-1]]))


def find_file(file_name):
    file_name = os.path.basename(file_name)
    for root, _, files in os.walk(PATH):
        if file_name in files:
            return os.path.join(root, file_name)
    raise FileNotFoundError("file {} do not exist".format(file_name))

def multiply_by_0_94(lr):
    return lr*0.94


# METRIC: BPC (bit per character)
def BPC(predictions, targets):
    cce = cross_entropy(predictions, targets, reduction='none')
    bpc = cce / torch.log(torch.tensor(2.))
    return bpc


def masked_BPC(mask_value):
    func = BPC
    return masked_function(func, mask_value)


def MaskedCrossEntropyLoss(mask_value):
    func = torch.nn.CrossEntropyLoss(reduce = False)
    return masked_function(func, mask_value)


def masked_function(func, mask_value, **kwargs):
    """ masked loss or metric computed on a batch of predictions / targets
    """

    mask_value = torch.tensor(mask_value, dtype=torch.float32)
    def ret_func(predictions, targets):
        targets = targets.view((-1,))
        predictions = predictions.view((-1, predictions.shape[-1]))
        mask = torch.eq(targets, mask_value)
        mask = 1 - mask.to(torch.float32)
        val = func(predictions, targets) * mask

        return torch.sum(val) / torch.sum(mask)
    return ret_func


class MaskedBPCMetric(Metric):
    def __init__(self, mask_value, **kwargs):
        super().__init__(**kwargs)
        self.metric_function = masked_BPC(mask_value)

    def __call__(self, predictions, targets):
        return self.metric_function(predictions, targets)
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        pass
    
    def compute(self) -> None:
        pass