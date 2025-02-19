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
File: layers.py
Created Date: Wed Mar 09 2022
Author: Randall Balestriero and Bobak Kiani
-----
Last Modified: Wed Mar 09 2022 10:02:19 PM
Modified By: Randall Balestriero
-----
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import sys
#sys.path.append("../../github/deel-torchlip/")
#sys.path.append("../../github/")

import torch
from torch import nn
import torch.nn.utils.parametrize as P
import utils
import numpy as np
from deel.torchlip import SpectralLinear
from quantized_layers import QLinear, QSpectralLinear #, QHadamard



# taken and modified from https://github.com/Lezcano/expRNN/blob/master/orthogonal.py
class modrelu(nn.Module):
    def __init__(self, features):
        # For now we just support square layers
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)
        # nn.init.zeros_(self.b.data)

    def forward(self, inputs):
        norm = torch.abs(inputs) + 1e-6
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sgn(inputs)

        return phase * magnitude

