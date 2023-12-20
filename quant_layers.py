"""
Customized quantization layers and modules

Example method:
SAWB-PACT: Accurate and Efficient 2-bit Quantized Neural Networks
RCF: Additive Power-of-Two Quantization: An Efficient Non-uniform Discretization For Neural Networks
"""
import torch
import torch.nn.functional as F
from torch import Tensor
#from .qexample import STE
from base import QBaseConv2d, QBase, QBaseLinear
import torch.nn as nn
#from new_quant import *
from qmethods import *

class SAWB(QBase):
    def __init__(self, nbit: int, train_flag: bool = True, qmode:str="symm"):
        super(SAWB, self).__init__(nbit, train_flag)
        self.register_buffer("alpha", torch.tensor(1.0))
        self.register_buffer("scale", torch.tensor(1.0))
        self.qmode = qmode

        # sawb
        z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}
        self.z = z_typical[f'{int(nbit)}bit']
    def q(self, input:Tensor):
        """
        Quantization method
        """
        m = input.abs().mean()
        std = input.std()

        if self.qmode == 'symm':
            n_lv = 2 ** (self.nbit - 1) - 1
            self.alpha.data = 1/self.z[0] * std - self.z[1]/self.z[0] * m
        elif self.qmode == 'asymm':
            n_lv = 2 ** (self.nbit - 1) - 1
            self.alpha.data = 2*m
        else:
            raise NotImplemented
    
        self.scale.data = n_lv / self.alpha
        
        if self.train_flag:
            xq = input.clamp(-self.alpha.item(), self.alpha.item())
            xq = xq.mul(self.scale).round()
            if len(xq.unique()) > 2**self.nbit:
                xq = xq.clamp(-2**self.nbit//2, 2**self.nbit//2-1)
            
            if self.dequantize:
                xq = xq.div(self.scale)
        else:
            xq = input
        return xq

    def trainFunc(self, input:Tensor):
        input = input.clamp(-self.alpha.item(), self.alpha.item())
        # get scaler
        _ = self.q(input)
        # quantization-aware-training
        out = STE.apply(input, self.scale.data)
        return out
    
    def evalFunc(self, input: Tensor):
        out = self.q(input)
        return out
        
class TernFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        tFactor = 0.05
        
        max_w = input.abs().max()
        th = tFactor*max_w #threshold
        output = input.clone().zero_()
        W = input[input.ge(th)+input.le(-th)].abs().mean()
        output[input.ge(th)] = W
        output[input.lt(-th)] = -W
        return output
    @staticmethod
    def backward(ctx, grad_output):
        # saved tensors - tuple of tensors with one element

        grad_input = grad_output.clone()
        return grad_input

class TernW(QBase):
    def __init__(self, nbit: int=2, train_flag: bool = True):
        super().__init__(nbit, train_flag)
        self.tFactor = 0.05
    
    def trainFunc(self, input: Tensor):
        out = TernFunc.apply(input)
        return out

class zero_quant(QBase):
    def __init__(self, nbit: int, train_flag: bool = True):
        super(zero_quant, self).__init__(nbit, train_flag)
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.tensor(1))
        self.register_buffer('zero', torch.tensor(1))
        self.nbit = nbit
        self.maxq = torch.tensor(2 ** self.nbit - 1)
        self.sym = True

    def forward(self, input:Tensor):
        x=input
        dev = x.device
        self.maxq = self.maxq.to(dev)
        shape = x.shape
        xf = x.flatten().unsqueeze(0)
        tmp = torch.zeros(xf.shape[0], device=dev)
        #tmp = torch.zeros(xf.shape[0])
        xmin = torch.minimum(xf.min(1)[0], tmp)
        xmax = torch.maximum(xf.max(1)[0], tmp)
        abs_min = xmin.abs_()
        xmax = torch.maximum(abs_min, xmax)
        tmp = xmin < 0
        if torch.any(tmp):
            xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale.data = xmax
          self.zero.data = xmin
        else:
          self.scale.data = (xmax - xmin) / self.maxq
          if self.sym:
              self.zero.data = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero.data = torch.round(-xmin / self.scale)
        x_r = input.div(self.scale).round() + self.zero
        q = torch.clamp(x_r, 0, self.maxq)
        out = self.scale * (q - self.zero)
        print(out.unique())
        return out
    # def forward(self, input:Tensor):
    #     out = input.mul(self.scale).add(self.bias)
    #     # Fused ReLU
    #     #out = F.relu(out)
    #     out = out.round()
    #     out = out.clamp(-self.nlv//2, self.nlv//2-1)
    #     out = out.div(self.scale)
    #     return out
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class MulQ(QBase):
    def __init__(self, nbit: int=2, train_flag: bool = True):
        super().__init__(nbit, train_flag)
    
    def trainFunc(self, input: Tensor):
        out = MulQuant.apply(input, fn=False)
        return out

class HardQuantizeConv(QBase):
    def __init__(self, nbit: int, train_flag: bool = True):
        super(HardQuantizeConv, self).__init__(nbit, train_flag)
        self.num_bits = nbit
        init_act_clip_val = 2.0
        self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]), requires_grad=True)
        self.zero = nn.Parameter(torch.Tensor([0]), requires_grad=False)

    def forward(self, x):
        gamma = (2**self.num_bits - 1)/(2**(self.num_bits - 1))
        scaling_factor = gamma * torch.mean(torch.mean(abs(x),dim=1,keepdim=True),dim=0,keepdim=True)
        scaling_factor = scaling_factor.detach()
        scaled_weights = x/scaling_factor
        cliped_weights = torch.where(scaled_weights < self.clip_val/2, scaled_weights, self.clip_val/2)
        cliped_weights = torch.where(cliped_weights > -self.clip_val/2, cliped_weights, -self.clip_val/2)
        n = float(2 ** self.num_bits - 1) / self.clip_val
        quan_weights_no_grad = scaling_factor * (torch.round((cliped_weights + self.clip_val/2) * n ) / n - self.clip_val/2)
        out = quan_weights_no_grad.detach() - scaled_weights.detach() + scaled_weights
        return out

class Clipping(QBase):
    def __init__(self, nbit: int, train_flag: bool = True):
        super(Clipping, self).__init__(nbit, train_flag)
        self.num_bits = nbit
        init_act_clip_val = 2.0
        self.l=0

    def forward(self, x):
        # #import pdb;pdb.set_trace()
        # self.l +=1
        # torch.save(x,'./activations/act'+str(self.l)+'.pt')
        out = x.clamp(min=-0.3,max=0.3)
        #out=x
        return out

class RCFNearest(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # define the range of q
        abit = 4
        nlv = torch.tensor(())
        for i in range(2**(abit-1)):
            nlv = torch.cat((nlv,torch.tensor([1/((2)**i)]).float()),0)
            nlv = torch.cat((nlv,-torch.tensor([1/((2)**i)]).float()),0)
        ulim = nlv.max()
        input_c = input.clamp(-ulim.item(), ulim.item())
        #sign = input_c.sign()
        input_q = nearest(input_c, nlv)
        ctx.save_for_backward(input, input_q, ulim)
        return input_q
    
    @staticmethod    
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()             # grad for weights will not be clipped
        input, input_q, ulim = ctx.saved_tensors
        i = (input.abs() > ulim).float()
        sign = input.sign()
        grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
        return grad_input

class Nearest(QBase):
    def __init__(self, nbit: int=2, train_flag: bool = True):
        super().__init__(nbit, train_flag)
    
    def trainFunc(self, input: Tensor):
        out = RCFNearest.apply(input)
        return out


class QConv2d(QBaseConv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, wbit: int = 32, abit: int = 32, train_flag=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, wbit, abit, train_flag)
        # layer index
        self.layer_idx = 0
        
        # quantizers
        if wbit < 32:
            if wbit in [4, 8]:
                self.wq = SAWB(self.wbit, train_flag=True, qmode="asymm")
            elif wbit in [2]:
                self.wq = TernW(train_flag=True)
        

    def forward(self, input:Tensor):
        wq = self.wq(self.weight)
        y = F.conv2d(input, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y

class QLinear(QBaseLinear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, wbit: int = 8, abit: int = 8, train_flag=True):
        super(QLinear, self).__init__(in_features, out_features, bias, wbit, abit, train_flag)

        shape=1
        # quantizers
        if wbit < 32:
            self.wq = SAWB(self.wbit, train_flag=True, qmode="asymm")
        if abit <= 32:
            self.aq = Nearest(self.abit)

    def trainFunc(self, input):
        return super().trainFunc(input)
