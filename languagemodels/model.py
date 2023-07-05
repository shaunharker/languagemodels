# Author: Shaun Harker
# Date: 2023-07-02
# License: MIT

import torch
from torch.nn import Module, Linear, Embedding, ModuleList
from torch.nn.functional import pad, softmax


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
    def __init__(self, n_vocab_out, d_model, n_layers):
        super().__init__()
        self.n_vocab_out = n_vocab_out
        self.d_model = d_model
        self.read_heads = ModuleList(Linear(d_model, n_vocab_out, bias=False) for _ in range(n_layers))

    def forward(self, x):
        # x.shape == (n_layers+1, batch_size, example_length, n_vocab_out)
        logits = torch.stack([self.read_heads[idx](x[idx]) for idx in range(x.shape[0])])
        probs = softmax(logits, dim=-1)
        return probs
    
    def add_layer(self):
        device = self.read_heads[0].weight.device
        self.read_heads.append(Linear(self.d_model, self.n_vocab_out, bias=False).to(device)) 


class LanguageModel(Module):
    def __init__(self,
                 n_vocab_in=256,
                 n_vocab_out=256,
                 bos=0,
                 module=None):
        super().__init__()
        self.text_input = TextInput(n_vocab_in=n_vocab_in, d_model=module.d_model, bos=bos)
        self.module = module
        self.text_output = TextOutput(n_vocab_out=n_vocab_out, d_model=module.d_model, n_layers=1+module.n_layers)
        
    def add_layer(self):
        self.module.add_layer()
        self.text_output.add_layer()

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
