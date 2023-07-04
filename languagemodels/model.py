# Author: Shaun Harker
# Date: 2023-07-02
# License: MIT

import math
import torch
from torch.nn import Module, Linear, Embedding, ModuleList, LayerNorm
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


class TransformerLayer(Module):
    def __init__(self, d_model, d_k, d_v, n_heads, mask="causal"):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        
        self.query_proj = Linear(d_model, d_k*n_heads, bias=False)
        self.key_proj = Linear(d_model, d_k*n_heads, bias=False)
        self.value_proj = Linear(d_model, d_v*n_heads, bias=False)
        self.mask = Mask(mask=mask)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.linear = Linear(d_v*n_heads, d_model, bias=False)

    def forward(self, x):
        n_ctx = x.shape[-2]
        split_heads = lambda x: x.view(x.shape[:-1]+(self.n_heads,self.d_k)).transpose(-2,-3).contiguous()
        merge_heads = lambda x: x.transpose(-2,-3).contiguous().view(x.shape[:-3]+(n_ctx,self.n_heads*self.d_v))
        (Q, K, V) = map(split_heads,(self.query_proj(x),self.key_proj(x),self.value_proj(x)))
        QKT = torch.matmul(Q/math.sqrt(self.d_k),K.transpose(-1,-2))
        return x + (self.linear(merge_heads(self.softmax(self.mask(QKT))@V)))


class Transformer(Module):
    def __init__(self,
                 d_model=1024,
                 d_k=64,
                 d_v=64,
                 n_heads=16,
                 n_layers=16):
        super().__init__()
        self.layers = ModuleList(TransformerLayer(d_model, d_k, d_v, n_heads) for _ in range(n_layers))

    def forward(self, x):
        xs = [x]
        for layer in self.layers:
            x = layer(x)
            xs.append(x)
        return torch.stack(xs, dim=0)


class TextInput(Module):
    def __init__(self, n_vocab_in, d_model, bos=0):
        super().__init__()
        self.bos = bos
        self.embedding = Embedding(n_vocab_in, d_model)

    def forward(self, input_ids):
        padded_input_ids = pad(input_ids, (1, 0), "constant", self.bos)
        x = self.embedding(padded_input_ids)
        return x


class TextOutput(Module):
    def __init__(self, n_vocab_out, d_model):
        super().__init__()
        self.linear = Linear(d_model, n_vocab_out, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.linear(x)
        probs = self.softmax(logits)
        return probs


class LanguageModel(Module):
    def __init__(self,
                 n_vocab_in=256,
                 n_vocab_out=256,
                 d_model=1024,
                 d_k=64,
                 d_v=64,
                 n_heads=16,
                 n_layers=16,
                 bos=0):
        super().__init__()
        self.text_input = TextInput(n_vocab_in=n_vocab_in, d_model=d_model, bos=bos)
        self.module = Transformer(d_model=d_model,
                                  d_k=d_k,
                                  d_v=d_v,
                                  n_heads=n_heads,
                                  n_layers=n_layers)
        self.text_output = TextOutput(n_vocab_out=n_vocab_out, d_model=d_model)
        
    def forward(self, input_ids):
        """
        Note: adds 1 to length, as it inserts a 0th column for bos token
        """
        x = self.text_input(input_ids)
        xs = self.module(x)
        probs = self.text_output(xs) # (n_layers+1, batch_size, example_length, n_vocab_out)
        return probs
    
    def losses(self, output_ids):
        input_ids = output_ids[...,:-1]
        probs = self.forward(input_ids) # (n_layers+1, batch_size, example_length, n_vocab_out)
        output_ids = torch.stack([output_ids] * probs.shape[0]).unsqueeze(-1) # (n_layers+1, batch_size, example_length, 1) uint8_t data (torch.long storage)
        return -torch.gather(probs, dim=-1, index=output_ids).log2() # cross entropy samples
