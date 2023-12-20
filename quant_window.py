import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
import sys
import os
sys.path("/home/ah2288/LP_MipNerF/qlib/quant_utils.py")
from quant_utils import *

class QuantAct(Module):
	"""
	Class to quantize given activations
	"""
	def __init__(self,
	             activation_bit,
	             full_precision_flag=False,
	             running_stat=True,
				 beta=0.9):
		"""
		activation_bit: bit-setting for activation
		full_precision_flag: full precision or not
		running_stat: determines whether the activation range is updated or froze
		"""
		super(QuantAct, self).__init__()
		self.activation_bit = activation_bit
		self.full_precision_flag = full_precision_flag
		self.running_stat = running_stat
		self.register_buffer('x_min', torch.zeros(1))
		self.register_buffer('x_max', torch.zeros(1))
		self.register_buffer('beta', torch.Tensor([beta]))
		self.register_buffer('beta_t', torch.ones(1))
		self.act_function = AsymmetricQuantFunction.apply
	
	def __repr__(self):
		return "{0}(activation_bit={1}, full_precision_flag={2}, running_stat={3}, Act_min: {4:.2f}, Act_max: {5:.2f})".format(
			self.__class__.__name__, self.activation_bit,
			self.full_precision_flag, self.running_stat, self.x_min.item(),
			self.x_max.item())
	
	def fix(self):
		"""
		fix the activation range by setting running stat
		"""
		self.running_stat = False
	
	def unfix(self):
		"""
		fix the activation range by setting running stat
		"""
		self.running_stat = True
	
	def forward(self, x):
		"""
		quantize given activation x
		"""

		if self.running_stat:
			x_min = x.data.min()
			x_max = x.data.max()
			# in-place operation used on multi-gpus
			# self.x_min += -self.x_min + min(self.x_min, x_min)
			# self.x_max += -self.x_max + max(self.x_max, x_max)

			self.beta_t = self.beta_t * self.beta
			self.x_min = (self.x_min * self.beta + x_min * (1 - self.beta))/(1 - self.beta_t)
			self.x_max = (self.x_max * self.beta + x_max * (1 - self.beta)) / (1 - self.beta_t)
	
		if not self.full_precision_flag:
			quant_act = self.act_function(x, self.activation_bit, self.x_min,
			                              self.x_max)
			return quant_act
		else:
			return x
