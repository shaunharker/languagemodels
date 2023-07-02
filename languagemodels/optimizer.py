import torch


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
            beta = self.param(self.n)
            self.x.mul_(beta).add_(x, alpha=1-beta)
        self.n += 1
        return self.x


class AdamW:
    """
    parameters: may be model.parameters() or model.named_parameters()
    """
    def __init__(self,
                 parameters,
                 lr=lambda n: 0.0 if n < 1000 else 1e-6,
                 beta1=lambda n: 0.9,
                 beta2=lambda n: 0.999,
                 weight_decay=lambda n: 0.001,
                 update=lambda n: True,
                 n=0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.update = update
        self.n = n

        try:
            self.parameters = dict(parameters)
        except:
            self.parameters = dict(enumerate(parameters))
        self.state = {
            name: {
                'lr': lr,
                'G': EMAFilter(beta1, init="zeros"),
                'G2': EMAFilter(beta2, init="zeros"),
                'weight_decay': weight_decay,
                'update': update,
                }
            for (name, p) in self.parameters.items()}
        for (name, p) in self.parameters.items():
            try:
                with torch.no_grad():
                    p.grad.data *= 0.0
            except:
                pass

    def update_parameters(self, parameters):
        try:
            self.parameters = dict(parameters)
        except:
            self.parameters = dict(enumerate(parameters))
        for (name, p) in self.parameters.items():
            if name not in self.state:
                self.state[name] = {
                    'lr': self.lr,
                    'G': EMAFilter(self.beta1, init="zeros"),
                    'G2': EMAFilter(self.beta2, init="zeros"),
                    'weight_decay': self.weight_decay,
                    'update': self.update}
                try:
                    p.grad.data *= 0.0
                except:
                    pass

    @torch.no_grad()
    def step(self, closure=None):
        with torch.enable_grad():
            if closure is not None:
                result = closure()
            else:
                result = None
        for (name, p) in self.parameters.items():
            n = self.n
            state = self.state[name]
            if state["update"](n):
                g = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                G = state['G'](g)
                G2 = state['G2'](torch.square(g))
                dp = G/torch.sqrt(G2)
                dp.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0).add_(p.data, alpha=state["weight_decay"](n))
                p.data.sub_(dp, alpha=state["lr"](n))
                p.grad.data *= 0.0
        self.n += 1
        return result


import torch
from torch.optim import Optimizer

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
        for group in self.param_groups:
            for p in group['params']:
                betas = group['betas']
                self.state[p] = {
                    'G': EMAFilter(betas[0], init="zeros"),
                    'G2': EMAFilter(betas[1], init="zeros")}
        self.zero_grad()

    def step(self, closure=None):
        self.n += 1
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                g = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                G = state['G'](g)
                G2 = state['G2'](torch.square(g))
                if self.n % group['batch_multiplier'] == 0:
                    dp = G/torch.sqrt(G2)
                    dp.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0).add_(p.data, alpha=group["weight_decay"])
                    p.data.sub_(dp, alpha=group["lr"])
        