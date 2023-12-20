# Author: Shaun Harker
# Date: 2023-07-02
# License: MIT

import torch
from torch.optim import Optimizer

class EMAFilter:
    def __init__(self, param, init="zeros"):
        self.param = param
        self.n = 0
        self.x = None
        self.init = init

    def __call__(self, x):
        if self.x is None:
            if self.init == "zeros":
                self.x = torch.zeros_like(x, memory_format=torch.preserve_format)
            elif self.init == "ones":
                self.x = torch.ones_like(x, memory_format=torch.preserve_format)
            else:
                raise ValueError(f"EMAFilter: unrecognized choice init = {self.init}")
        else:
            if self.x.shape != x.shape:
                new_x = torch.zeros_like(x, memory_format=torch.preserve_format)
                slices = [slice(0, dim) for dim in self.x.shape]
                new_x[slices] = self.x
                self.x = new_x
            beta = self.param
            self.x.mul_(beta).add_(x, alpha=1-beta)
        self.n += 1
        return self.x


class CustomAdamW(Optimizer):
    def __init__(self, params, lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01, batch_multiplier=16):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, batch_multiplier=batch_multiplier)
        super().__init__(params, defaults)
        self.n = 0
        self.state = {}
        self.zero_grad()

    def step(self, closure=None):
        self.n += 1
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p not in self.state:
                    self.state[p] = {'G': EMAFilter(group['betas'][0], init="zeros"), 
                                     'G2': EMAFilter(group['betas'][1], init="zeros")}
                state = self.state[p]
                g = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                G = state['G'](g)
                G2 = state['G2'](torch.square(g))
                dp = G/torch.sqrt(G2)
                dp.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0).add_(p.data, alpha=group["weight_decay"])
                p.data.sub_(dp, alpha=group["lr"])
        