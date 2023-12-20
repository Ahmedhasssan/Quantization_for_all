"""
Post Training Quantization
"""

import torch
from torch import Tensor
from .base import QBase
import torch.nn as nn

def round_ste(x: Tensor):
    """
    Straight through estimator
    """
    return (x.round() - x).detach() + x

def lp_loss(pred, target, p=2.0, reduction='none'):
    """
    loss function measured in lp norm
    """
    if reduction == 'none':
        return (pred-target).abs().pow(p).sum(1).mean()
    else:
        return (pred-target).abs().pow(p).mean()

class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class AdaRound(QBase):
    """
    Weight quantizer: Up or Down? Adaptive Rounding for Post-Training Quantization
    
    https://arxiv.org/abs/2004.10568
    """
    def __init__(self, nbit: int = 8, train_flag: bool=True, weights: torch.Tensor=None) -> None:
        super().__init__(nbit)
        self.iter = 0
        self.train_flag = train_flag
        self.weight_int=0

        self.register_buffer("lb", weights.min())
        self.register_buffer("ub", weights.max())

        # integer boundary
        self.lower = (-(1 << (self.nbit-1)))
        self.upper = ((1 << (self.nbit-1)) - 1)

        self.qlb = 0
        self.qub = 2**self.nbit - 1

        # initialize the alpha
        self.init_flag = True

        # parameters
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3

        # register the learnable parameter
        self.register_alpha(weights)

    def register_alpha(self, x:torch.Tensor):
        xfloor = x.div(self.scale).floor()

        # compute alpha
        diff = x.div(self.scale).sub(xfloor)
        alpha = -torch.log((self.zeta-self.gamma) / (diff - self.gamma) - 1)
        self.register_parameter("alpha", torch.nn.Parameter(alpha))

    def get_qparam(self, x:torch.Tensor):
        lb = torch.min(self.lb, x.min())
        ub = torch.max(self.ub, x.max())

        # update boundary
        self.lb = lb.clone()
        self.ub = ub.clone()

    def h(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    
    def q(self, x:torch.Tensor):
        scale = self.ub.sub(self.lb).div(self.qub - self.qlb)
        offset = self.qlb - torch.round(self.lb.mul(scale))

        self.scale.copy_(scale)
        self.offset.copy_(offset)

        if self.init_flag:
            self.register_alpha(x)
            self.init_flag = False

        # quantization
        xfloor = x.div(self.scale).floor()
        soft_shift = self.h()

        # quantize
        if self.train_flag:
            xint = xfloor + soft_shift
        else:
            xint = xfloor + self.alpha.ge(0.0).float()
            self.weight_int=xint

        xq = xint + self.offset
        out = torch.clamp(xq, self.lower, self.upper)
        #### Get the values here
        # dequantize
        out = out.sub(self.offset).mul(self.scale)
        return out
    
    def trainFunc(self, x: torch.Tensor):
        self.get_qparam(x)
        return super().trainFunc(x)

    def evalFunc(self, x: torch.Tensor):
        xq = self.q(x)
        return xq

class LearningQ(QBase):
    """
    Activation quantizer with fully learnable scaling factor

    The learnable scaling factor is initialized as the optimal value of the first batch
    """
    def __init__(self, nbit: int = 8, train_flag: bool = True):
        super().__init__(nbit)
        self.train_flag = train_flag

        # register learnable parameter 
        self.register_parameter("delta", torch.nn.Parameter(torch.tensor(1.0)))
        
        # initialization flag
        self.initialize = False

        self.prev_int_frame = None
        self.round_err = 0.0
        self.frame_err_all = []

    def compute_frame_err(self, xq:torch.Tensor):
        if self.prev_int_frame is None:
            self.prev_int_frame = xq
        else:
            self.round_err = xq.sub(self.prev_int_frame).abs()
            self.prev_int_frame = xq

            self.frame_err_all.append(self.round_err)
    
    def get_fp_range(self, x:torch.Tensor):
        y = torch.flatten(x, start_dim=1)
        batch_min = torch.min(y, 1)[0].mean()
        batch_max = torch.max(y, 1)[0].mean()
        return batch_min, batch_max
    
    def quantize(self, x:torch.Tensor, xmin, xmax):
        delta = (xmax - xmin) / (2 ** self.nbit - 1)
        zero_point = (-xmin / delta).round()

        xint = torch.round(x / delta)
        xq = torch.clamp(xint + zero_point, 0, 2**self.nbit - 1)
        xdq = (xq - zero_point) * delta
        return xdq
    
    def initialize_qparam(self, x:torch.Tensor):
        """
        Find the optimal scaling factor in the first batch
        """
        x_min, x_max = self.get_fp_range(x)
        best_loss = 1e+10

        for i in range(80):
            new_max = x_max * (1.0 - (i * 0.01))
            new_min = x_min * (1.0 - (i * 0.01))

            # quantize and dequantize for mse 
            xdq = self.quantize(x, new_min, new_max)
            loss = lp_loss(xdq, x, p=2.4, reduction='all')

            if loss < best_loss:
                best_loss = loss
                delta = (new_max - new_min) / (2**self.nbit - 1)
                zero_point = (-new_min / delta).round()
        
        return delta, zero_point

    def q(self, x:torch.Tensor):
        if not self.initialize:
            if self.train_flag:
                delta, zero_point = self.initialize_qparam(x)
                self.delta.data = delta
                self.offset.data = zero_point
                self.initialize = True

        # quantize
        xr = round_ste(x / self.delta)
        xq = torch.clamp(xr, min=-2**(self.nbit-1), max=2**(self.nbit-1)-1)

        # dequantize
        xdq = (xq) * self.delta
        
        return xdq
    
class LearnigQ_old(QBase):
    """
    Activation quantizer with fully learnable scaling factor

    The learnable scaling factor is initialized as the optimal value of the first batch
    """
    def __init__(self, nbit: int = 8, train_flag: bool = True):
        super().__init__(nbit)
        self.train_flag = train_flag

        # register learnable parameter 
        self.register_parameter("delta", torch.nn.Parameter(torch.tensor(1.0)))
        
        # initialization flag
        self.initialize = False

        self.prev_int_frame = None
        self.round_err = 0.0
        self.frame_err_all = []

    def compute_frame_err(self, xq:torch.Tensor):
        if self.prev_int_frame is None:
            self.prev_int_frame = xq
        else:
            self.round_err = xq.sub(self.prev_int_frame).abs()
            self.prev_int_frame = xq

            self.frame_err_all.append(self.round_err)
    
    def get_fp_range(self, x:torch.Tensor):
        y = torch.flatten(x, start_dim=1)
        batch_min = torch.min(y, 1)[0].mean()
        batch_max = torch.max(y, 1)[0].mean()
        return batch_min, batch_max
    
    def quantize(self, x:torch.Tensor, xmin, xmax):
        delta = (xmax - xmin) / (2 ** self.nbit - 1)
        zero_point = (-xmin / delta).round()

        xint = torch.round(x / delta)
        xq = torch.clamp(xint + zero_point, 0, 2**self.nbit - 1)
        xdq = (xq - zero_point) * delta
        return xdq
    
    def initialize_qparam(self, x:torch.Tensor):
        """
        Find the optimal scaling factor in the first batch
        """
        x_min, x_max = self.get_fp_range(x)
        best_loss = 1e+10

        for i in range(80):
            new_max = x_max * (1.0 - (i * 0.01))
            new_min = x_min * (1.0 - (i * 0.01))

            # quantize and dequantize for mse 
            xdq = self.quantize(x, new_min, new_max)
            loss = lp_loss(xdq, x, p=2.4, reduction='all')

            if loss < best_loss:
                best_loss = loss
                delta = (new_max - new_min) / (2**self.nbit - 1)
                zero_point = (-new_min / delta).round()
        
        return delta, zero_point

    def q(self, x:torch.Tensor):
        if not self.initialize:
            if self.train_flag:
                delta, zero_point = self.initialize_qparam(x)
                self.delta.data = delta
                self.offset.data = zero_point

        # quantize
        xr = round_ste(x / self.delta) + self.offset
        xq = torch.clamp(xr, min=0.0, max=2**self.nbit-1)

        # dequantize
        xdq = (xq - self.offset) * self.delta

        if not self.train_flag:
            self.compute_frame_err(xdq)
        
        return xdq

class FixedGridQ_Cov(QBase):
    def __init__(self, nbit: int = 8, train_flag: bool = True):
        super().__init__(nbit)
        self.train_flag = train_flag
        self.register_parameter("delta", torch.nn.Parameter(torch.tensor(1.0)))
        self.initialize = True
        self.nbit = nbit
        self.l=0
        self.nlv = torch.Tensor([])
    
    def initialize_qparam(self, x:torch.Tensor):
        x_min, x_max = self.get_fp_range(x)
        best_loss = 1e+10

    def q(self, x:torch.Tensor):
        input = x
        abit = self.nbit
        if self.initialize:
            for i in range(2**(abit)-1):
                # self.nlv = torch.cat((self.nlv,torch.tensor([1/((2)**i)]).float()),0)
                # self.nlv = torch.cat((self.nlv,torch.tensor([(1+i)/((2)**i)]).float()),0)
                self.nlv = torch.cat((self.nlv,torch.tensor([((1+i)/((2)**18))]).float()),0)
                self.nlv = torch.cat((self.nlv,torch.tensor([((1+i)/((2)**19))]).float()),0)
                self.nlv = torch.cat((self.nlv,torch.tensor([((1+i)/((2)**21))]).float()),0)
            self.nlv = torch.cat((self.nlv,torch.tensor([0.])),0)
            self.initialize =False
        ulim = input.max()
        input_c = input.clamp(0, ulim.item())
        input_q = nearest(input_c, self.nlv)
        #print(len(input_q.unique()))
        return input_q
    
class FixedGridQ_U(QBase):
    def __init__(self, nbit: int = 8, train_flag: bool = True):
        super().__init__(nbit)
        self.train_flag = train_flag
        self.register_parameter("delta", torch.nn.Parameter(torch.tensor(1.0)))
        self.initialize = True
        self.nbit = nbit
        self.l=0
        self.nlv = torch.Tensor([])
    
    def initialize_qparam(self, x:torch.Tensor):
        x_min, x_max = self.get_fp_range(x)
        best_loss = 1e+10

    def q(self, x:torch.Tensor):
        input = x
        abit = self.nbit
        if self.initialize:
            for i in range(2**(abit)-1):
                self.nlv = torch.cat((self.nlv,torch.tensor([2/((2)**i)]).float()),0)
                self.nlv = torch.cat((self.nlv,torch.tensor([(2*(1+i)/((2**i)))]).float()),0)
                self.nlv = torch.cat((self.nlv,-torch.tensor([2/((2)**i)]).float()),0)
                self.nlv = torch.cat((self.nlv,-torch.tensor([(2*(1+i)/((2**i)))]).float()),0)
                self.nlv = torch.cat((self.nlv,torch.tensor([((1+i)/((2)**5))]).float()),0)
                self.nlv = torch.cat((self.nlv,-torch.tensor([((1+i)/((2)**5))]).float()),0)
                self.nlv = torch.cat((self.nlv,torch.tensor([((i)/((2)**6))]).float()),0)
                self.nlv = torch.cat((self.nlv,-torch.tensor([((i)/((2)**6))]).float()),0)
                self.nlv = torch.cat((self.nlv,torch.tensor([((i)/((2)**7))]).float()),0)
                self.nlv = torch.cat((self.nlv,-torch.tensor([((i)/((2)**7))]).float()),0)
            self.nlv = torch.cat((self.nlv,torch.tensor([0.])),0)
            self.initialize =False
        ulim = input.max()
        input_c = input.clamp(-ulim.item(), ulim.item())
        input_q = nearest(input_c, self.nlv)
        #print(len(input_q.unique()))
        return input_q

def nearest(input:torch.Tensor, lv:torch.Tensor):
    r"""
    Round the high precision input the nearest level
    """
    shape = input.shape
    xq = input.view(-1)
    
    value_s = lv.type_as(input)
    idxs = (xq.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]  # project to nearest quantization level
    xq = value_s[idxs].view(shape)
    return xq
    
class STE(torch.autograd.Function):
    """
    Straight through estimator
    """
    @staticmethod
    def forward(ctx, input, scale):
        input_q = input.mul(scale).round()
        out = input_q.div(scale)
        ctx.save = input
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()*(ctx.input>0)
        return grad_input, None
    
class SAWB(QBase):
    def __init__(self, nbit: int, train_flag: bool = True, qmode:str="asymm"):
        super(SAWB, self).__init__(nbit, train_flag)
        self.register_buffer("alpha", torch.tensor(1.0))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_parameter("delta", torch.nn.Parameter(torch.tensor(1.0)))
        self.qmode = qmode
        self.initialize = False
        self.act=0

        # sawb
        z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114], '32bit':[0.027, 1.114]}
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
            n_lv = 2 ** (self.nbit-1)
            self.alpha.data = 3*m
        else:
            raise NotImplemented
    
        self.scale.data = n_lv / self.alpha
        
        if self.train_flag:
            xq = input.clamp(0, self.alpha.item())
            xq = xq.mul(self.scale).round()
            self.act=xq
            if len(xq.unique()) > 2**self.nbit:
                xq = xq.clamp(-2**self.nbit//2, 2**self.nbit//2-1)
            if self.dequantize:
                xq = xq.div(self.scale)
        else:
            xq = input
        return xq

    def trainFunc(self, input:Tensor):
        input = input.clamp(0, self.alpha.item())
        # get scaler
        _ = self.q(input)
        # quantization-aware-training
        out = STE.apply(input, self.scale.data)
        return out
    
    def evalFunc(self, input: Tensor):
        out = self.q(input)
        return out

class BatchQuant(QBase):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned=True, momentum=0.9):
        super().__init__(nbit, train_flag)
        
        # statistical based scaling factor
        self.register_buffer("delta", torch.tensor(1.0))

        # running boundary
        self.register_buffer("running_min", torch.tensor(0.0))
        self.register_buffer("running_max", torch.tensor(0.0))
        self.momentum = momentum

        if unsigned:
            self.qlb = 0.0
            self.qub = 2**(self.nbit) - 1
        else:
            self.qlb = -2**(self.nbit - 1)
            self.qub = 2**(self.nbit - 1) - 1

    def get_qparam(self, x:Tensor):
        # average the min max value along the batch dim
        y = torch.flatten(x, start_dim=1)
        batch_min = torch.min(y, 1)[0].min()
        batch_max = torch.max(y, 1)[0].max()

        # if batch_min.abs() > self.running_min.abs():
        #     self.running_min = batch_min
        
        # if batch_max.abs() > self.running_max.abs():
        #     self.running_max = batch_max

        # if self.running_min == 0:
        #     self.running_min = batch_min
        #     self.running_max = batch_max
        # else:
        #     self.running_min = self.running_min.mul(self.momentum) + batch_min
        #     self.running_max = self.running_max.mul(self.momentum) + batch_max

        ub = max(batch_min, batch_max)
        scale = ub / ((self.qub - self.qlb) / 2)
        zero_point = self.qlb - batch_min.div(scale).round()

        # update scale
        self.delta.copy_(scale)
        self.offset.copy_(zero_point)

        return scale, zero_point

    def q(self, x:Tensor):
        xq = x.div(self.delta) + self.offset
        xq = xq.clamp(self.qlb, self.qub)
        
        xq = Round.apply(xq)
        xq = xq.sub(self.offset)
        
        if self.dequantize:
            xq = xq * self.delta
        return xq

    def trainFunc(self, input: Tensor):
        self.get_qparam(input)
        xq = self.q(input)

        # update the final scaling factor
        self.scale.copy_(self.delta)
        return xq
    
    def evalFunc(self, input: Tensor):
        xq = input.mul(self.scale) + self.offset
        xq = xq.clamp(self.qlb, self.qub).round()
        xq = xq.sub(self.offset)
        return xq
    
######## GENIE ######
#####################
def uniform_quantize(x, delta, zero_point, n_bits):
    x_int = torch.round(x / delta)
    x_q = torch.clamp(x_int + zero_point, 0, 2**n_bits - 1)
    x_deq = (x_q-zero_point) * delta
    return x_deq

def init_scale(x, n_bits, symmetric, channel_wise, signed=True):
    # parallel batch
    n_batch = x.shape[0] if channel_wise else 1
    x_flat = x.reshape(n_batch, -1).detach()

    best_score = torch.full([n_batch], 1e+10, device=x.device)

    # Four cases need to be considered: {signed, unsigned} x {symmetric, asymmetric}
    if symmetric:
        max_value = x_flat.abs().max(dim=1).values
        x_max = max_value
        x_min = -max_value if signed else torch.zeros_like(x_max)
    else:
        x_max = x_flat.max(dim=1).values
        x_min = x_flat.min(dim=1).values if signed else torch.max(x_flat.min(dim=1).values, torch.tensor([0.]))

    delta = torch.zeros_like(best_score)
    zero_point = torch.zeros_like(best_score)

    # Finding scales
    for clip_ratio in torch.arange(1.0, 0.0, -0.01):
        new_max, new_min = x_max * clip_ratio, x_min * clip_ratio

        new_delta = (new_max-new_min) / (2**n_bits - 1)
        new_zeropoint = (- new_min/new_delta).round()
        x_q = uniform_quantize(x_flat, new_delta.unsqueeze(1), new_zeropoint.unsqueeze(1), n_bits)
        score = (x_flat-x_q).abs().pow(2.4).mean(dim=1)

        delta = torch.where(score < best_score, new_delta, delta)
        zero_point = torch.where(score < best_score, new_zeropoint, zero_point)
        best_score = torch.minimum(score, best_score)

    if torch.any(delta < 1e-10):
        log.warning(f'Quantization range close to zero: [{delta}]')

    target_dim = [-1, *[1]*(len(x.shape)-1)]
    return delta.view(target_dim), zero_point.view(target_dim)
    
def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x

class ActivationQuantizer(nn.Module):
    """
    An implementation of the Learned Step Size Quantization and QDrop.
    References:
    https://arxiv.org/pdf/1902.08153.pdf
    https://arxiv.org/abs/2203.05740.pdf
    """

    def __init__(self, nbit:int):
        super(ActivationQuantizer, self).__init__()
        self.register_buffer("delta", torch.tensor(1.0))
        self.n_bits = nbit
        self.scale = None
        self.initialized = False
        self.signed = None
        self.train_mode = False
        self.initialize = False

    def forward(self, x):
        if not self.initialized:
            self.signed = x.min() < 0
            self.scale = nn.Parameter(
                init_scale(x, n_bits=self.n_bits, symmetric=True, channel_wise=False, signed=self.signed)[0])
            self.initialized = True

        Qn = - 2**(self.n_bits-1) if self.signed else 0
        Qp = 2**(self.n_bits-1) - 1 if self.signed else 2**self.n_bits - 1

        v, s = x, self.scale
        v_bar = round_ste(torch.clamp(v / s, Qn, Qp))
        v_hat = v_bar * s

        # use QDrop
        if self.train_mode:
            return torch.where(torch.rand_like(x) < 0.5, v_hat, x)
        else:
            return v_hat

    def extra_repr(self) -> str:
        return f'n_bits={self.n_bits}, signed={self.signed}, scale={self.scale}'
