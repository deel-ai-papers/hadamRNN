import torch
import torch.nn.utils.parametrize as P
from torch import nn
from torch.nn.utils import parametrize
from quantized_layers import QLinear, _Quantize, quantize, QSpectralLinear, QSpectralLinearNoBjorck
import numpy as np

class _Quantize_stats(_Quantize):
    def __init__(
        self,
        weight: torch.Tensor,
        num_bits: int = 8, 
        enforce_true_quantized: bool = True,
        delta_m: torch.Tensor = None,
        max_absolute: float = 0.,
        pre_scaling_factor: float = 1.,
    ) -> None:
        super().__init__(weight,num_bits,enforce_true_quantized,delta_m)
        self.max_absolute = max_absolute
        self.stat_max_absolute = 1.e-10
        self.pre_scaling_factor = pre_scaling_factor
    def compute_stats(self, min_v, max_v) -> None:
            self.stat_max_absolute = np.max([np.abs(min_v),np.abs(max_v), self.stat_max_absolute])
    def compute_quantize(self, weight: torch.Tensor, min_v : float, max_v: float) -> torch.Tensor:
        #print(weight.shape, min_v, max_v)
        if self.delta_m is not None:
            self.delta_m = self.delta_m.to(weight.device)
            qweight = quantize(weight-self.delta_m, num_bits=self.num_bits,
                                    min_value=min_v,
                                    max_value=max_v,
                                    enforce_true_quantized=self.enforce_true_quantized) \
                        + self.delta_m
        else:
            qweight = quantize(weight, num_bits=self.num_bits,
                                    min_value=min_v,
                                    max_value=max_v,
                                    enforce_true_quantized=self.enforce_true_quantized)
        return qweight
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        weight = weight * self.pre_scaling_factor
        if self.max_absolute == 0.:
            min_v, max_v = float(weight.min()), float(weight.max())
            self.compute_stats(min_v, max_v)
            return self.compute_quantize(weight, min_v, max_v)
        else:
            #print("use self.max_absolute", self.max_absolute)
            return self.compute_quantize(weight, self.max_absolute, self.max_absolute)  # keep learnt value

class QSSM(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, activation=None, activation_final=None, bias=True, bias_final=True, num_bits=0, manytomany=False, seed=None, pre_scaling_feat=1.0, num_bits_feat= None, qoutput = False, **kwargs):
		if seed: torch.manual_seed(seed)
		super(QSSM, self).__init__()
		activation_final = None
		self.hidden_size = hidden_size
		self.activation = activation
		self.activation_final = activation_final
		self.manytomany = manytomany
		self.input_layer = QLinear(input_size, hidden_size, bias=False, num_bits=None)
		self.recurrent_layer = QLinear(hidden_size, hidden_size, bias=bias, num_bits=num_bits)
		if qoutput:
			self.output_layer = QLinear(hidden_size, output_size, bias=bias_final, num_bits=None)
		else:
			self.output_layer = nn.Linear(hidden_size, output_size, bias=bias_final)
		self.quant_input = None
		self.quant_feat = None
		if num_bits_feat is not None:
			self.quant_input = _Quantize_stats(weight = None, num_bits=num_bits_feat)
			self.quant_feat = _Quantize_stats(weight = None, num_bits=num_bits_feat,pre_scaling_factor=pre_scaling_feat)


	def default_hidden(self, input):
		return input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

	def forward(self, inputs):
		with P.cached():
			hidden = self.default_hidden(inputs)
			outputs = []
			for input in torch.unbind(inputs, dim=1):
				if self.quant_feat is not None:
					input = self.quant_input(input)
					hidden = self.quant_feat(hidden)
				hidden = self.recurrent_layer(hidden) + self.input_layer(input)
				if self.manytomany:
					if self.activation is not None: 
						outputs.append(self.output_layer(self.activation(hidden)))
					else:
						outputs.append(self.output_layer(hidden))
		if self.manytomany: return torch.stack(outputs,dim=1).view(-1,len(outputs),self.output_layer.out_features)
		if self.activation is not None:
			output = self.output_layer(self.activation(hidden))
		else:
			output = self.output_layer(hidden)
		if self.activation_final is not None: output = self.activation_final(output)
		return output

class SpectralQSSM(QSSM):

	def __init__(self, input_size, hidden_size, output_size, activation=None, activation_final=None, bias=True, bias_final=True, num_bits=0, manytomany=False, use_bjorck=False, seed=None, **kwargs):
		#super(BinactivadamRNN, self).__init__(self, input_size, hidden_size, output_size, activation=activation, activation_final=activation_final, bias=bias, bias_final=bias_final, num_bits=num_bits, manytomany=manytomany, basic_block=basic_block, **kwargs)
		super(SpectralQSSM, self).__init__(input_size, hidden_size, output_size, activation, activation_final, bias, bias_final, num_bits, manytomany, seed, **kwargs)
		if use_bjorck:
			self.recurrent_layer = QSpectralLinear(hidden_size, hidden_size, bias, num_bits)
		else:
			self.recurrent_layer = QSpectralLinearNoBjorck(hidden_size, hidden_size, bias, num_bits)
	

class QSSMwithEmbeddings(QSSM):
	

	def __init__(self, input_size, hidden_size, output_size, activation=None, activation_final=None, bias=True, bias_final=True, num_bits=0, manytomany=False, seed=None, pre_scaling_feat=1.0, num_bits_feat= None, qoutput = False, **kwargs):
		super(QSSMwithEmbeddings, self).__init__(input_size, hidden_size, output_size, activation, activation_final, bias, bias_final, num_bits, manytomany, seed, pre_scaling_feat, num_bits_feat, qoutput, **kwargs)
		self.embedding = True
		self.word_embedding = nn.Embedding(10000, input_size, padding_idx=0)
		
	def forward(self, inputs):
		inputs = self.word_embedding(inputs)
		return super().forward(inputs)
	