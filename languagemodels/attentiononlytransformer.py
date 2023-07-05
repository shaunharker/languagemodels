# Author: Shaun Harker
# Date: 2023-07-02
# License: MIT

import torch
from torch.nn import Module, Linear, ModuleList
from torch.nn.functional import softmax

    
class MultiHeadAttentionLayer(Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.proj = Linear(d_model, 3*d_model, bias=False)

    def forward(self, x):
        device = x.device
        n_ctx = x.shape[-2]
        split_heads = lambda x: x.view(x.shape[:-1] + (self.n_heads, -1)).transpose(-2, -3).contiguous()
        merge_heads = lambda x: x.transpose(-2, -3).contiguous().view(x.shape[:-3] + (n_ctx, -1))
        Q, K, V = map(split_heads, torch.split(self.proj(x), self.d_model, dim=-1))
        mask = (1 - 1 / torch.tril(torch.ones((n_ctx, n_ctx), device=device)))
        QKT = torch.matmul(Q, K.transpose(-1, -2)) + mask
        x = x + merge_heads(softmax(QKT, dim=-1) @ V)
        return x

class AttentionOnlyTransformer(Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        self.n_layers = self.config.pop('n_layers')
        self.layers = ModuleList(MultiHeadAttentionLayer(**config) for _ in range(self.n_layers))

    def forward(self, x):
        xs = [x]
        for layer in self.layers:
            x = layer(x)
            xs.append(x)
        return torch.stack(xs, dim=0)
    
    def add_layer(self):
        device = self.layers[0].proj.weight.device
        self.layers.append(MultiHeadAttentionLayer(**self.config).to(device))
        self.n_layers += 1