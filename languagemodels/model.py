# Author: Shaun Harker
# Date: 2023-07-02
# License: MIT

import math
import torch
from torch.nn import Module, Linear, Embedding, ModuleList, LayerNorm, GELU
from torch.nn.functional import pad


class Mask(Module):
    def __init__(self, mask="none"):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        n, device = x.shape[-1], x.device
        if self.mask == "none":
            return x
        elif self.mask == "causal":
            weight = (1-1/torch.tril(torch.ones((n,n),device=device)))
            return x + weight


class Attn(Module):
    def __init__(self, d_in, d_out, d_k, d_v, n_heads, mask="causal"):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        
        self.mask = Mask(mask=mask)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.query_proj = Linear(d_in, d_k*n_heads, bias=True)
        self.key_proj = Linear(d_in, d_k*n_heads, bias=True)
        self.value_proj = Linear(d_in, d_v*n_heads, bias=True)
        self.linear = Linear(d_v*n_heads, d_out, bias=True)

    def forward(self, x):
        n_ctx = x.shape[-2]
        split_heads = lambda x: x.view(x.shape[:-1]+(self.n_heads,self.d_k)).transpose(-2,-3).contiguous()
        merge_heads = lambda x: x.transpose(-2,-3).contiguous().view(x.shape[:-3]+(n_ctx,self.n_heads*self.d_v))
        (Q, K, V) = map(split_heads,(self.query_proj(x),self.key_proj(x),self.value_proj(x)))
        QKT = torch.matmul(Q/math.sqrt(self.d_k),K.transpose(-1,-2))
        return self.linear(merge_heads(self.softmax(self.mask(QKT))@V))


class TransformerLayer(Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_hidden):
        super().__init__()
        self.attn = Attn(d_model, d_model, d_k, d_v, n_heads, 'causal')
        self.ln1 = LayerNorm(d_model)
        self.ff1 = Linear(d_model, d_hidden)
        self.nonlinearity = GELU()
        self.ff2 = Linear(d_hidden, d_model)
        self.ln2 = LayerNorm(d_model)
        self.mlp = lambda x: self.ff2(self.nonlinearity(self.ff1(x)))
    def forward(self, x):
        return x + self.ln2(self.mlp(x + self.ln1(self.attn(x))))


class Transformer(Module):
    def __init__(self,
                 d_model=1024,
                 d_k=64,
                 d_v=64,
                 n_heads=16,
                 d_hidden=4096,
                 n_layers=16):
        super().__init__()
        self.layers = ModuleList(TransformerLayer(d_model, d_k, d_v, n_heads, d_hidden) for _ in range(n_layers))

    def forward(self, x):
        xs = [x]
        for layer in self.layers:
            x = layer(x)
            xs.append(x)
        return torch.stack(xs, dim=0)


class TextInput(Module):
    def __init__(self, n_vocab_in, d_model, bos=0):
        super().__init__()
        self.embedding = Embedding(n_vocab_in, d_model)
        self.bos = bos

    def forward(self, input_ids):
        padded_input_ids = pad(input_ids, (1, 0), "constant", self.bos)
        x = self.embedding(padded_input_ids)
        return x


class LinearAutoregression(Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = Linear(d_model, d_model, bias=False)
        self.linear.weight.data *= 0.0

    def forward(self, x):
        xs = []
        n_ctx = x.shape[-1]
        for idx in range(n_ctx):
            if idx == 0:
                xs.append(x[...,idx,:])
            else:
                xs.append(x[...,idx,:] + self.linear(xs[-1]))
        return torch.stack(xs,dim=-2)


class TextOutput(Module):
    def __init__(self, n_vocab_out, d_model):
        super().__init__()
        self.linear = Linear(d_model, n_vocab_out)

    def forward(self, x):
        return self.linear(x)
    

class LanguageModel(Module):
    def __init__(self,
                 n_vocab_in=256,
                 n_vocab_out=256,
                 d_model=1024,
                 d_k=64,
                 d_v=64,
                 n_heads=16,
                 d_hidden=4096,
                 n_layers=16,
                 bos=0):
  
        super().__init__()
        self.text_input = TextInput(n_vocab_in=n_vocab_in, d_model=d_model, bos=bos)
        self.module = Transformer(d_model=d_model,
                                  d_k=d_k,
                                  d_v=d_v,
                                  n_heads=n_heads,
                                  d_hidden=d_hidden,
                                  n_layers=n_layers)
        self.text_output = TextOutput(n_vocab_out=n_vocab_out, d_model=d_model)
        self.softmax = torch.nn.Softmax(dim=-1)


    def forward(self, input_ids):
        """
        Note: adds 1 to length, as it inserts a 0th column for bos token
        """
        x = self.text_input(input_ids)
        xs = self.module(x)
        logits = self.text_output(xs) # logits.shape == (n_layers+1, batch_size, example_length, n_vocab_out)
        probs = self.softmax(logits)  # (n_layers+1, batch_size, example_length, n_vocab_out)
        return probs
    
    def losses(self, output_ids):
        input_ids = output_ids[...,:-1]
        probs = self.forward(input_ids) # (n_layers+1, batch_size, example_length, n_vocab_out)
        output_ids = torch.stack([output_ids] * probs.shape[0]).unsqueeze(-1) # (n_layers+1, batch_size, example_length, 1) uint8_t data (torch.long storage)
        return -torch.gather(probs, dim=-1, index=output_ids).log2() # cross entropy samples
