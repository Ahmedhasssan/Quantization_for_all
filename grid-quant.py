"""
Low precision function for low precision gaussian
"""

import torch as th
import torch.nn.functional as thf


class GridQuant(th.nn.Module):
    def __init__(self, interval:float) -> None:
        super().__init__()

        self.interval = th.tensor(interval)
        self.initialize = True
        self.levels = None

    def round2grid(self, x:th.Tensor, value_s):
        assert self.levels is not None, "The grid is not properly initialized!"

        shape = x.shape
        xhard = x.view(-1)
        value_s = value_s.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
        xhard = value_s[idxs].view(shape)
        return xhard

    def forward(self, x:th.Tensor) -> th.Tensor:
        x=x.clamp(-2,2)
        if self.initialize:
            ub = x.max().round()
            lb = x.min().round()
            qrange = ub.sub(lb).abs()
            self.levels = th.tensor([lb + self.interval*i for i in range(int(qrange / self.interval))])
            self.initialize = False

        # round to nearest grid level
        output = self.round2grid(x, self.levels)
        return output
    
class GridQuant_fix(th.nn.Module):
    def __init__(self, nbit, lb=-1.0, ub=1.0) -> None:
        super().__init__()

        self.initialize = True
        self.levels = None
        self.qrange = ub-lb
        self.interval = (ub-lb)/(2**nbit)
        self.levels = th.tensor([lb + self.interval*i for i in range(int(self.qrange / self.interval))])

    def round2grid(self, x:th.Tensor, value_s):
        assert self.levels is not None, "The grid is not properly initialized!"
        

        shape = x.shape
        xhard = x.view(-1)
        value_s = value_s.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
        xhard = value_s[idxs].view(shape)
        return xhard

    def forward(self, x:th.Tensor) -> th.Tensor:
        # round to nearest grid level
        output = self.round2grid(x, self.levels)
        return output
        
