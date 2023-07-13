# Author: Shaun Harker
# Date: 2023-07-02
# License: MIT

import math
import torch
from torch.nn import Module, Linear, LayerNorm, GELU
from torch.nn.functional import softmax

class TransformerLayer(Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_hidden, sdp=True, init_scale=1.0):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        
        self.split_heads = lambda x: x.view(x.shape[:-1]+(self.n_heads,self.d_k)).transpose(-2,-3).contiguous()
        self.merge_heads = lambda x: x.transpose(-2,-3).contiguous().view(x.shape[:-3]+(-1,self.n_heads*self.d_v))

        self.query_proj = Linear(d_model, d_k*n_heads, bias=True)
        self.key_proj = Linear(d_model, d_k*n_heads, bias=True)
        self.value_proj = Linear(d_model, d_v*n_heads, bias=True)
        self.linear = Linear(d_v*n_heads, d_model, bias=False)
        self.ln1 = LayerNorm(d_model)

        self.ff1 = Linear(d_model, d_hidden, bias=True)
        self.nonlinearity = GELU()
        self.ff2 = Linear(d_hidden, d_model, bias=False)
        self.ln2 = LayerNorm(d_model)
        self.ln1.weight.data *= init_scale
        self.ln2.weight.data *= init_scale
        self.sdp = sdp

    def forward(self, x, layer_data=None):
        (Q, Kf, Vf) = map(self.split_heads,(self.query_proj(x),self.key_proj(x),self.value_proj(x)))
        if layer_data is None:
            K = Kf
            V = Vf
        else:
            Ks = torch.split(Kf, 1, dim=-2)
            Vs = torch.split(Vf, 1, dim=-2)
            layer_data.extend(zip(Ks, Vs))
            K = torch.cat([KV[0] for KV in layer_data], dim=-2)
            V = torch.cat([KV[1] for KV in layer_data], dim=-2)

        if self.sdp:
            attn = torch.nn.functional.scaled_dot_product_attention(query=Q, key=K, value=V, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            device = x.device
            n_ctx = x.shape[-2]
            mask = (1-1/torch.tril(torch.ones((n_ctx,n_ctx),device=device)))
            QKT = torch.matmul(Q,K.transpose(-1,-2)) + mask
            attn = softmax(QKT, dim=-1)@V

        x = x + self.ln1(self.linear(self.merge_heads(attn)))
        x = x + self.ln2(self.ff2(self.nonlinearity(self.ff1(x))))

        return x

class PersephoneLayer(Module):
    def __init__(self, d_model, d_k, init_scale=1.0):
        super().__init__()
        self.ln = LayerNorm(d_model)
        self.proj = Linear(d_model, d_k, bias=True)

    def forward(self, x, layer_data=None):
        self.proj(x)
        # idea. everybody makes a key. then we sum the keys from our d_model//d_k predecessors.
        # we could use FTC: 
        #   F(b)-F(a) = sum f(i), which lets us do cumsum differences.
        #   However, I worry about "subtraction of close numbers" issues with numerics.
        #   A sequential iterative approach is more precise, but it is slow.

        x = x + self.ln(x)
        return x
    

class MultiHeadAttentionLayer(Module):
    def __init__(self, d_model, n_heads, init_scale=1.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.proj = Linear(d_model, 3*d_model, bias=True)
        self.proj.weight.data *= init_scale
        self.proj.bias.data *= init_scale

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


class ExperimentalLayer1(Module):
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
    

class ExperimentalLayer2(Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.proj = Linear(d_model, 2*d_model, bias=False)
        self.proj.weight.data *= 0

    def forward(self, x): # 3 * d_model**2 * n_ctx + d_model * n_ctx ** 2
        device = x.device
        n_ctx = x.shape[-2]
        split_heads = lambda x: x.view(x.shape[:-1] + (self.n_heads, -1)).transpose(-2, -3).contiguous()
        merge_heads = lambda x: x.transpose(-2, -3).contiguous().view(x.shape[:-3] + (n_ctx, -1))
        Q, K, V = map(split_heads, (x,) + torch.split(self.proj(x), self.d_model, dim=-1))
        mask = (1 - 1 / torch.tril(torch.ones((n_ctx, n_ctx), device=device)))
        QKT = torch.matmul(Q, K.transpose(-1, -2)) + mask
        x = x + merge_heads(softmax(QKT, dim=-1) @ V)
        return x
    

class ExperimentalLayer3(Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.proj = Linear(d_model, d_model, bias=False)
        self.proj.weight.data *= 0

    def forward(self, x):
        device = x.device
        n_ctx = x.shape[-2]
        split_heads = lambda x: x.view(x.shape[:-1] + (self.n_heads, -1)).transpose(-2, -3).contiguous()
        merge_heads = lambda x: x.transpose(-2, -3).contiguous().view(x.shape[:-3] + (n_ctx, -1))
        Q = split_heads(x)
        K = split_heads(self.proj(x))
        mask = (1 - 1 / torch.tril(torch.ones((n_ctx, n_ctx), device=device)))
        QKT = torch.matmul(Q, K.transpose(-1, -2)) + mask
        x = x + merge_heads(softmax(QKT, dim=-1) @ Q)
        return x
    

class ExperimentalLayer4(Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.proj = Linear(d_model, d_model, bias=False)
        self.proj.weight.data *= 0

    def forward(self, x):
        device = x.device
        n_ctx = x.shape[-2]
        split_heads = lambda x: x.view(x.shape[:-1] + (self.n_heads, -1)).transpose(-2, -3).contiguous()
        merge_heads = lambda x: x.transpose(-2, -3).contiguous().view(x.shape[:-3] + (n_ctx, -1))
        Q = split_heads(x)  # (batch_size, n_heads, n_ctx, d_model//n_heads)
        K = split_heads(self.proj(x))
        mask = (1 - 1 / torch.tril(torch.ones((n_ctx, n_ctx), device=device)))
        QKT = torch.matmul(Q, K.transpose(-1, -2)) + mask
        x = x + merge_heads(softmax(QKT, dim=-1) @ K)
        return x
    
class ExperimentalLayer5(Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj = Linear(d_model, d_model, bias=False)
        self.proj.weight.data *= 0

    def forward(self, x):
        n_ctx = x.shape[-2]
        mask = (1 - 1 / torch.tril(torch.ones((n_ctx, n_ctx), device=x.device)))
        K = self.proj(x)
        x = x + (softmax(torch.matmul(x, K.transpose(-1, -2)) + mask, dim=-1) @ K)
        return x

class ExperimentalLayer6(Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj = Linear(d_model, 3*d_model, bias=False)
        self.proj.weight.data *= 0

    def forward(self, x):
        device = x.device
        n_ctx = x.shape[-2]
        Q, K, V = torch.split(self.proj(x), self.d_model, dim=-1)
        mask = (1 - 1 / torch.tril(torch.ones((n_ctx, n_ctx), device=device)))
        QKT = torch.matmul(Q, K.transpose(-1, -2)) + mask
        x = x + (softmax(QKT, dim=-1) @ V)
        return x
    
class ExperimentalLayer7(Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj = Linear(d_model, 2*d_model, bias=False)
        self.proj.weight.data *= 0

    def forward(self, x):
        device = x.device
        n_ctx = x.shape[-2]
        K, V = torch.split(self.proj(x), self.d_model, dim=-1)
        mask = (1 - 1 / torch.tril(torch.ones((n_ctx, n_ctx), device=device)))
        QKT = torch.matmul(x, K.transpose(-1, -2)) + mask
        x = x + (softmax(QKT, dim=-1) @ V)
        return x

class ExperimentalLayer8(Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.proj = Linear(d_model, 2*d_model, bias=True)
        self.proj.weight.data *= 0
        self.proj.bias.data *= 0

    def forward(self, x): # 3 * d_model**2 * n_ctx + d_model * n_ctx ** 2
        device = x.device
        n_ctx = x.shape[-2]
        split_heads = lambda x: x.view(x.shape[:-1] + (self.n_heads, -1)).transpose(-2, -3).contiguous()
        merge_heads = lambda x: x.transpose(-2, -3).contiguous().view(x.shape[:-3] + (n_ctx, -1))
        Q, K, V = map(split_heads, (x,) + torch.split(self.proj(x), self.d_model, dim=-1))
        mask = (1 - 1 / torch.tril(torch.ones((n_ctx, n_ctx), device=device)))
        QKT = torch.matmul(Q, K.transpose(-1, -2)) + mask
        x = x + merge_heads(softmax(QKT, dim=-1) @ V)
        return x
    
class ExperimentalLayer9(Module):
    def __init__(self, d_model, n_heads, d_hidden):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.key_proj = Linear(d_model, d_model, bias=True)
        self.value_proj = Linear(d_model, d_hidden, bias=True)
        self.nonlinearity = GELU()
        self.ff = Linear(d_hidden, d_model, bias=True)

    def forward(self, x):
        device = x.device
        n_ctx = x.shape[-2]
        split_heads = lambda x: x.view(x.shape[:-1] + (self.n_heads, -1)).transpose(-2, -3).contiguous()
        merge_heads = lambda x: x.transpose(-2, -3).contiguous().view(x.shape[:-3] + (n_ctx, -1))
        Q, K, V = map(split_heads, (x, self.key_proj(x), self.value_proj(x)))
        mask = (1 - 1 / torch.tril(torch.ones((n_ctx, n_ctx), device=device)))
        QKT = torch.matmul(Q, K.transpose(-1, -2)) + mask
        x = x + self.ff(self.nonlinearity(merge_heads(softmax(QKT, dim=-1) @ V)))
        return x
    
class ExperimentalLayer10(Module):
    def __init__(self, d_model, n_heads, d_hidden, init_scale=1.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.query_proj = Linear(d_model, d_model, bias=True)
        self.key_proj = Linear(d_model, d_model, bias=True)
        self.value_proj = Linear(d_model, d_hidden, bias=False)
        self.ff1 = Linear(d_model, d_hidden, bias=True)
        self.nonlinearity = GELU()
        self.ff2 = Linear(d_hidden, d_model, bias=False)
        self.ln = LayerNorm(d_model)
        self.ln.weight.data *= init_scale


    def forward(self, x):
        device = x.device
        n_ctx = x.shape[-2]
        split_heads = lambda x: x.view(x.shape[:-1] + (self.n_heads, -1)).transpose(-2, -3).contiguous()
        merge_heads = lambda x: x.transpose(-2, -3).contiguous().view(x.shape[:-3] + (n_ctx, -1))
        Q, K, V = map(split_heads, (self.query_proj(x), self.key_proj(x), self.value_proj(x)))
        
        # mask = (1 - 1 / torch.tril(torch.ones((n_ctx, n_ctx), device=device)))
        # QKT = torch.matmul(Q, K.transpose(-1, -2)) + mask
        # attn = softmax(QKT, dim=-1) @ V

        attn = torch.nn.functional.scaled_dot_product_attention(query=Q, key=K, value=V, attn_mask=None, dropout_p=0.0, is_causal=True)

        x = x + self.ln(self.ff2(self.nonlinearity(self.ff1(x) + merge_heads(attn))))
        return x