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
import torch
import torch.nn.utils.parametrize as P
from torch import nn

import numpy as np
import layers
import utils
#from layers import LSTMCell
from quantized_layers import QLinear, Binadamard, BinaryNorm #, projnetRecurrent, QSpectralLinear, QSpectralLinearNoBjorck
from deel.torchlip import SpectralLinear


class QRNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, activation=None, activation_final=None, bias=True, bias_final=True, num_bits=0, manytomany=False, seed=None, **kwargs):
		if seed: torch.manual_seed(seed)
		super(QRNN, self).__init__()
		activation_final = None
		self.hidden_size = hidden_size
		self.activation = activation
		self.activation_final = activation_final
		self.manytomany = manytomany
		self.input_layer = QLinear(input_size, hidden_size, bias=False, num_bits=None)
		self.recurrent_layer = QLinear(hidden_size, hidden_size, bias=bias, num_bits=num_bits)
		self.output_layer = nn.Linear(hidden_size, output_size, bias=bias_final)

	def default_hidden(self, input):
		return input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

	def forward(self, inputs):
		with P.cached():
			hidden = self.default_hidden(inputs)
			outputs = []
			for input in torch.unbind(inputs, dim=1):
				hidden = self.recurrent_layer(hidden) + self.input_layer(input)
				if self.activation is not None: hidden = self.activation(hidden)
				if self.manytomany:
					outputs.append(self.output_layer(hidden))
		if self.manytomany: return torch.stack(outputs,dim=1).view(-1,self.output_layer.out_features)
		output = self.output_layer(hidden)
		if self.activation_final is not None: output = self.activation_final(output)
		return output



class BinadamSSM(nn.Module):
	""" 1 or 2-layers binary hadamard SSM (linear recurrence) with relu activation in between

		This network has only ternary (-1 / 0 / 1) weights. The first layer is block-diagonal with
		blocks of size 2**(basic_block) and with weights -1/1. The second layer is fully connected
		(the kernel is made of only -1 / 1 weights)
	"""

	def __init__(self, input_size, hidden_size, output_size, activation=None, activation_final=None, bias=True, bias_final=True, num_bits=0, manytomany=False, basic_block=None, seed=None, qoutput = False, single_layer = False, **kwargs):
		if seed: torch.manual_seed(seed)
		super(BinadamSSM, self).__init__()
		self.hidden_size = hidden_size
		self.activation = activation
		self.activation_final = activation_final
		self.manytomany = manytomany
		self.input_layer = QLinear(input_size, hidden_size, bias=False, num_bits=num_bits)
		#self.input_layer_2 = Binadamard(hidden_size, hidden_size, bias=False, basic_block="full")
		self.recurrent_layer = Binadamard(hidden_size, hidden_size, bias=bias, basic_block=basic_block)
		if single_layer:
			self.input_layer_2 = None
			self.recurrent_layer_2 = None
		else:
			self.input_layer_2 = BinaryNorm(hidden_size, hidden_size)
			self.recurrent_layer_2 = Binadamard(hidden_size, hidden_size, bias=bias, basic_block="full")
		if qoutput:
			self.output_layer = QLinear(hidden_size, output_size, bias=bias_final, num_bits=num_bits)
		else:
			self.output_layer = nn.Linear(hidden_size, output_size, bias=bias_final)
		self.dropout = None

	def default_hidden(self, input):
		return input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

	def forward(self, inputs):
		with P.cached():
			h1 = self.default_hidden(inputs)
			h2 = self.default_hidden(inputs)
			outputs = []
			for input in torch.unbind(inputs, dim=1):
				h1 = self.recurrent_layer(h1) + self.input_layer(input)
				if self.activation is not None: h1b = self.activation(h1)
				if self.input_layer_2 is not None:
					h2 = self.recurrent_layer_2(h2) + self.input_layer_2(h1b)				
					if self.activation is not None: h2b = self.activation(h2)
				else:
					h2 = h1 #self.recurrent_layer_2(h2) + self.input_layer_2(h1b)
					h2b = h1b
				if self.dropout is not None:
					h2b = self.dropout(h2b)
				if self.manytomany:
					outputs.append(self.output_layer(h2b))
		if self.manytomany: 
			outputs = torch.stack(outputs,dim=1).view(-1,len(outputs),self.output_layer.out_features)
			return outputs #torch.mean(outputs, dim=1)
		#if self.manytomany: return outputs
		output = self.output_layer(h2b)
		if self.activation_final is not None: output = self.activation_final(output)
		return output
	
	def initialize_gradients(self):
		self.recurrent_layer.sign_vector.requires_grad = False
		self.recurrent_layer_2.sign_vector.requires_grad = False
		self.input_layer_2.original_weight.requires_grad = False

	def alternate_gradients(self):
		for _, p in self.named_parameters():
			if p.requires_grad: p.requires_grad = False
			else: p.requires_grad = True

	def apply_weight_constraints_(self):
		self.recurrent_layer.apply_weight_constraints_()
		self.recurrent_layer_2.apply_weight_constraints_()


class BinadamSSMwithEmbeddings(BinadamSSM):
	
	def __init__(self, input_size, hidden_size, output_size, activation=None, activation_final=None, bias=True, bias_final=True, num_bits=0, manytomany=False, basic_block=None, seed=None, qoutput = False, single_layer = False, vocab_size = 10000, **kwargs):
		super(BinadamSSMwithEmbeddings, self).__init__(input_size, hidden_size, output_size, activation, activation_final, bias, bias_final, num_bits, manytomany, basic_block, seed, qoutput, single_layer, **kwargs)
		self.embedding = True
		self.word_embedding = nn.Embedding(vocab_size, input_size, padding_idx=0)
		dropout = 0.
		self.dropout = None
		if dropout > 0:
			self.dropout = nn.Dropout(dropout)
		
	def forward(self, inputs):
		inputs = self.word_embedding(inputs)
		if self.dropout is not None:
			inputs = self.dropout(inputs)
		return super().forward(inputs)
	


