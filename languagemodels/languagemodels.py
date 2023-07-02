import math
import torch
from torch.nn import Module, Linear, Embedding, ModuleList, LayerNorm, GELU
from torch.nn.functional import pad
import numpy as np
from .dataset import FastPileBytesDataset
import time
import asyncio
import torch
from torch.optim import AdamW
from IPython.display import display, HTML


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
        self.d_in = d_in
        self.d_out = d_out
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
        (n_ctx, d_model) = x.shape[-2:]
        assert d_model == self.d_in, f"{d_model} != {self.d_in}"
        split_heads = lambda x: x.view(x.shape[:-1]+(self.n_heads,-1)).transpose(-2,-3).contiguous()
        merge_heads = lambda x: x.transpose(-2,-3).contiguous().view(x.shape[:-3]+(n_ctx,self.d_v*self.n_heads))
        (Q, K, V) = map(split_heads,(self.query_proj(x),self.key_proj(x),self.value_proj(x)))
        QKT = torch.matmul(Q/math.sqrt(self.d_k),K.transpose(-1,-2))
        return self.linear(merge_heads(self.softmax(self.mask(QKT))@V))


class TransformerLayer(Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_hidden):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
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
                 n_vocab_in=256,
                 n_vocab_out=256,
                 d_model=1024,
                 d_k=64,
                 d_v=64,
                 n_heads=16,
                 d_hidden=4096,
                 n_layers=16):
        super().__init__()
        self.n_vocab_in = n_vocab_in
        self.n_vocab_out = n_vocab_out
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.n_layers = n_layers

        self.layers = ModuleList(TransformerLayer(d_model, d_k, d_v, n_heads, d_hidden) for _ in n_layers)


    def forward(self, x):
        xs = [x]
        for layer in self.layers:
            x = layer(x)
            xs.append(x)
        return torch.stack(xs, dim=0)


    def get_config(self):
        return {
            'n_vocab_in': self.n_vocab_in,
            'n_vocab_out': self.n_vocab_out,
            'd_model': self.d_model,
            'd_k': self.d_k,
            'd_v': self.d_v,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers
        }  


    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.get_config()
        }, f=path)

    @staticmethod
    def load(path):
        checkpoint = torch.load(path)
        config = checkpoint['config']
        sd = checkpoint['state_dict']
        module = Transformer(**config)
        module.load_state_dict(sd)
        return module


class TextInput(Module):
    def __init__(self, n_vocab_in, d_model):
        self.n_vocab_in = n_vocab_in
        self.d_model = d_model
        self.linear = Linear(d_model, d_model, bias=False)
        self.linear.weight.data *= 0.0
        self.embedding = Embedding(n_vocab_in, d_model)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        n_ctx = input_ids.shape[-1]
        for idx in range(n_ctx-1):
            x[...,idx+1,:] += self.linear(x[...,idx,:])
        return x


class TextOutput(Module):
    def __init__(self, n_vocab_out, d_model):
        self.n_vocab_out = n_vocab_out
        self.d_model = d_model
        self.linear = Linear(d_model, n_vocab_out)

    def forward(self, x):
        return self.linear(x)
    

class LanguageModel(Module):
    def __init__(self,
                 n_vocab_in=256,
                 n_vocab_out=256,
                 n_ctx=512,
                 d_model=1024,
                 d_k=64,
                 d_v=64,
                 n_heads=16,
                 d_hidden=4096,
                 n_layers=16,
                 bos=0):
        """
        Unused bytes in utf-8 encodings:
        [0, 2, 4, 5, 6, 11, 14, 15, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 127, 192, 193, 222, 223, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]
        """
        super().__init__()
        self.n_vocab_in = n_vocab_in
        self.n_vocab_out = n_vocab_out
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.bos = bos  # beginning of sequence token
        self.text_input = TextInput(n_vocab_in=n_vocab_in, d_model=d_model)
        self.module = Transformer(n_vocab_in=n_vocab_in,
                                  n_vocab_out=n_vocab_out,
                                  n_ctx=n_ctx,
                                  d_model=d_model,
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
        padded_input_ids = pad(input_ids, (1, 0), "constant", self.bos)
        x = self.text_input(padded_input_ids)
        xs = self.module(x)
        logits = self.text_output(xs) # logits.shape == (n_layers+1, batch_size, example_length, n_vocab_out)
        probs = self.softmax(logits)  # (n_layers+1, batch_size, example_length, n_vocab_out)
        return probs
    
    def losses(self, output_ids):
        input_ids = output_ids[...,:-1]
        probs = self.forward(input_ids) # (n_layers+1, batch_size, example_length, n_vocab_out)
        output_ids = torch.stack([output_ids] * (self.n_layers+1)) # (n_layers+1, batch_size, example_length) uint8_t data (torch.long storage)
        return -torch.gather(probs, dim=-1, index=output_ids).log2() # cross entropy samples

class Trainer:
    def __init__(self,
                 example_length=512,
                 batch_size=1,
                 batch_multiplier=8):
        """
        example_length: length of examples to use for training
        batch_size: number of examples in a single batch
        batch_multiplier: number of batches to accumulate gradients on
                          before calling optimizer.step. This makes for
                          an effective batch_size of batch_size*batch_multiplier,
                          hence the name.
        """
        self.example_length = example_length
        self.batch_size = batch_size
        self.batch_multiplier = batch_multiplier
        self.dataset = FastPileBytesDataset(example_length=512)
        self.model = LanguageModel()
        self.optimizer = AdamW(self.model.named_parameters())
        self.inbox = []
        self.loss_by_layer = []
        self.last_layer_loss = []
        self.times = []
        self.start_time = time.time()
        self.n = 0
        self.D = 0 # total data

    def clear_stats(self):
        self.inbox = []
        self.loss_by_layer = []
        self.last_layer_loss = []
        self.times = []
        self.n = 0
        self.D = 0 # total data

    def train(self):
        def closure():
            if len(self.inbox) > 0:
                batch = self.inbox.pop()
            if batch is None:
                batch = self.dataset.batch(batch_size=self.batch_size, example_length=self.example_length)
            self.D += self.batch_size * self.example_length
            losses = self.model.losses(batch)
            losses = torch.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0)
            loss = torch.mean(losses)
            batch_loss_by_layer = np.mean(losses.detach().cpu().numpy(), axis=(1,2)) # over bs and pos
            batch_last_layer_loss_by_pos = np.mean(losses[-1].detach().cpu().numpy(), axis=0) # over bs
            batch_last_layer_loss = np.mean(batch_last_layer_loss_by_pos)
            loss.backward()
            return batch_loss_by_layer, batch_last_layer_loss
        if self.n % self.batch_multiplier == 0:
            batch_loss_by_layer, batch_last_layer_loss = self.optimizer.step(closure)
            self.optimizer.zero_grad()
        else:
            batch_loss_by_layer, batch_last_layer_loss = closure()
        self.n += 1
        self.loss_by_layer.append(batch_loss_by_layer)
        self.last_layer_loss.append(batch_last_layer_loss)
        self.times.append(time.time())

    def autocomplete(self,
                     prompt=None,
                     n_ctx=None,
                     temp=1.0,
                     n_generate=512,
                     output_layer=-1):
        Categorical = torch.distributions.Categorical
        if n_ctx is None:
            n_ctx = 512
        if prompt is None:
            prompt ="""In a shocking finding, scientists discovered a herd of unicorns living in a
    remote, previously unexplored valley in the Andes Mountains. Even more
    surprising to the researchers was the fact that the unicorns spoke perfect
    English."""
        if device is None:
            device = self.model.device
        n_vocab_out = self.n_vocab_out
        input_ids = self.dataset.encode(prompt)
        input_ids = input_ids[-n_ctx:]
        prompt = self.dataset.decode(input_ids)

        async def sampler():
            for _ in range(n_generate):
                await asyncio.sleep(0)  # Give other tasks a chance to run
                probs = self.model(torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0))[output_layer].view(-1)[-n_vocab_out:]
                if temp > 0:
                    y = Categorical(probs=probs**(1.0/temp)).sample().item()
                else:
                    y = torch.argmax(probs).item()
                input_ids = (input_ids + [y])[-n_ctx:]
                yield y

        return sampler
    
    async def display_autocomplete(self,
                     prompt=None,
                     n_ctx=None,
                     temp=1.0,
                     n_generate=512,
                     output_layer=-1):
        
        sampler = self.autocomplete(prompt, n_ctx, temp, n_generate, output_layer)
        display_handle = display(HTML(prompt), display_id=True)
        list_of_tokens = []
        async for c in sampler():
            list_of_tokens.append(c)
            completion = self.dataset.decode(list_of_tokens)
            def cleanup(s):
                return s.replace('<', '&lt;').replace('>', '&gt;')
            contents = ('<pre style="background: black; color: lime; font-family: monospace">'+
                    '<span style="color: white">' + cleanup(prompt) + '</span>' + cleanup(completion) + '\n'*20 + '</pre>')
            display_handle.update(HTML(contents))
        return display_handle
