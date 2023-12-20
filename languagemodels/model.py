# Author: Shaun Harker
# Date: 2023-07-02
# License: MIT

import torch
from torch.nn import Module, Linear, Embedding, ModuleList, GELU, LayerNorm
from torch.nn.functional import pad, softmax, one_hot
import inspect
from IPython.display import display, HTML
import asyncio
from torch.utils.checkpoint import checkpoint
import random
from pathlib import Path
import math

import types
import numpy as np
import os

def utf8encode(char_sequence):
    if type(char_sequence) == types.GeneratorType:
        def stream():
            for c in char_sequence:
                for b in bytes(c, encoding='utf8'):
                    yield b
        result = stream()
    else:
        result = bytes(char_sequence, encoding='utf8')
    return result

def utf8decode(byte_sequence):
    def is_valid_utf8_byte(b):
        return b&0b11111000 != 0b11111000
    def is_payload_utf8_byte(b):
        return b&0b11000000 == 0b10000000
    def is_header_utf8_byte(b):
        return is_valid_utf8_byte(b) and not is_payload_utf8_byte(b)
    def char_width(b):
        if b&0b10000000 == 0:
            return 1
        elif b&0b11100000 == 0b11000000:
            return 2
        elif b&0b11110000 == 0b11100000:
            return 3
        elif b&0b11111000 == 0b11110000:
            return 4
        return None
    def stream():
        (word, width) = ([], 0)
        for b in byte_sequence:
            if is_header_utf8_byte(b):
                (word, width) = ([b], char_width(b))
            elif is_payload_utf8_byte(b):
                word.append(b)
            if len(word) == width:
                try:
                    yield bytes(word).decode('utf8')
                except:
                    # There are still undecodables we catch here.
                    # e.g. bytes(map(lambda x: int(x,base=2),['0b11000000', '0b10000000'])).decode('utf8') raises UnicodeDecodeError
                    pass
    if type(byte_sequence) == types.GeneratorType:
        return stream()
    else:
        return ''.join(list(stream()))
        
# """
# Unused bytes in utf-8 encodings:
# [0, 2, 4, 5, 6, 11, 14, 15, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 127, 192, 193, 222, 223, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]
# """

class TextInput(Module):
    def __init__(self, n_vocab_in, d_model, bos=0, **kwargs):
        super().__init__()
        self.bos = bos
        self.n_vocab_in = n_vocab_in
        self.d_model = d_model
        assert n_vocab_in <= d_model

    def forward(self, input_ids):
        return one_hot(pad(input_ids, (1, 0), "constant", self.bos), self.d_model).float()

    def insert_layer(self, index, **kwargs):
        return []
    
    def append_layer(self, **kwargs):
        return []

    def prepend_layer(self, **kwargs):
        return []


class TextOutput(Module):
    def __init__(self, n_vocab_out, d_model, **kwargs):
        super().__init__()
        self.n_vocab_out = n_vocab_out
        self.d_model = d_model

    def forward(self, x):
        logits = x[..., -self.n_vocab_out:]
        probs = softmax(logits, dim=-1)
        return probs
    
    def insert_layer(self, index, **kwargs):
        return []
    
    def append_layer(self, **kwargs):
        return []

    def prepend_layer(self, **kwargs):
        return []


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
        is_causal = True
        if layer_data is None:
            (Q, K, V) = map(self.split_heads,(self.query_proj(x),self.key_proj(x),self.value_proj(x)))
        else:
            # in this mode, layer_data has (K, V), and x has suffix for which we haven't made predictions yet
            if len(layer_data) == 2:
                is_causal = False
                (Ki, Vi) = layer_data
                (Q, Kf, Vf) = map(self.split_heads,(self.query_proj(x),self.key_proj(x),self.value_proj(x)))
                K = torch.cat([Ki, Kf], dim=-2)
                V = torch.cat([Vi, Vf], dim=-2)
                layer_data[0] = K
                layer_data[1] = V
            elif len(layer_data) == 0:
                (Q, K, V) = map(self.split_heads,(self.query_proj(x),self.key_proj(x),self.value_proj(x)))
                layer_data.append(K)
                layer_data.append(V)
            else:
                raise ValueError(f"layer_data corrupt: {layer_data}")

        if self.sdp:
            attn = torch.nn.functional.scaled_dot_product_attention(query=Q, key=K, value=V, attn_mask=None, dropout_p=0.0, is_causal=is_causal)
        else:
            device = x.device
            n_ctx = x.shape[-2]
            mask = (1-1/torch.tril(torch.ones((n_ctx,n_ctx),device=device)))
            QKT = torch.matmul(Q,K.transpose(-1,-2)) + mask
            attn = softmax(QKT, dim=-1)@V

        x = x + self.ln1(self.linear(self.merge_heads(attn)))
        x = x + self.ln2(self.ff2(self.nonlinearity(self.ff1(x))))

        return x


class LanguageModel(Module):
    def __init__(self,
                 n_vocab_in=256,
                 n_vocab_out=256,
                 bos=0,
                 d_model=1024,
                 n_layers=16,
                 input_class=None,
                 input_config=None,
                 input_classname=None,
                 input_classcode=None,
                 output_class=None,
                 output_config=None,
                 output_classname=None,
                 output_classcode=None,
                 layer_class=None,
                 layer_config=None,
                 layer_classname=None,
                 layer_classcode=None,
                 random_layers=False,
                 checkpointing=False):
        super().__init__()
        self.n_vocab_in = n_vocab_in
        self.n_vocab_out = n_vocab_out
        self.bos = bos
        self.d_model = d_model
        self.n_layers = n_layers
        self.random_layers = random_layers  # shuffle layers each forward pass
        self.depth_limit = n_layers
        self.checkpointing = checkpointing

        def load_class(code, classname):
            local_vars = {}
            exec(code, {'torch': torch,
                        'Module': Module,
                        'ModuleList': ModuleList,
                        'Linear': Linear,
                        'Embedding': Embedding,
                        'LayerNorm': LayerNorm,
                        'GELU': GELU,
                        'pad': pad,
                        'softmax': softmax,
                        'one_hot': one_hot}, local_vars)
            return local_vars[classname]

        # input_class
        if input_class is None:
            try:
                input_class = load_class(input_classcode, input_classname)
            except:
                input_class = TextInput
        try:
            self.input_classname = input_class.__name__
        except:
            self.input_classname = input_classname
        
        try:
            self.input_classcode = inspect.getsource(input_class)
        except:
            self.input_classcode = input_classcode

        self.input_class = input_class

        ## default input config
        if input_config is None:
            input_config = {'n_vocab_in': n_vocab_in, 'bos': bos, 'd_model': d_model}
        
        self.input_config = input_config # if input_config is not None else {}

        # output_class
        if output_class is None:
            try:
                output_class = load_class(output_classcode, output_classname)
            except:
                output_class = TextOutput
        try:
            self.output_classname = output_class.__name__
        except:
            self.output_classname = output_classname
        
        try:
            self.output_classcode = inspect.getsource(output_class)
        except:
            self.output_classcode = output_classcode

        self.output_class = output_class

        ## default output config
        if output_config is None:
            output_config = {'n_vocab_out': n_vocab_out, 'd_model': d_model, 'n_layers': n_layers}

        self.output_config = output_config # if output_config is not None else {}

        # layer_class
        if layer_class is None:
            layer_class = load_class(layer_classcode, layer_classname)

        self.layer_class = layer_class

        try:
            self.layer_classname = layer_class.__name__
        except:
            self.layer_classname = layer_classname
        
        try:
            self.layer_classcode = inspect.getsource(self.layer_class)
        except:
            self.layer_classcode = layer_classcode
            
        self.layer_config = layer_config if layer_config is not None else {}
        
        self.text_input = input_class(**self.input_config)
        self.layers = ModuleList(layer_class(**self.layer_config) for _ in range(self.n_layers))
        self.text_output = output_class(**self.output_config)

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except:
            return 'cpu'

    def insert_layer(self, index):
        self.layers.insert(index, self.layer_class(**self.layer_config).to(self.device))
        params = list(self.layers[index].parameters())
        params += self.text_input.insert_layer(index)
        params += self.text_output.insert_layer(index)
        self.n_layers += 1
        if 'n_layers' in self.output_config:
            self.output_config['n_layers'] += 1
            # TODO refactor how I handle configs for appending, serialization, extensibility considerations
        return params
    
    def append_layer(self):
        return self.insert_layer(index=self.n_layers)
    
    def prepend_layer(self):
        return self.insert_layer(index=0)

    def copy_layer(self, src_idx, trg_idx):
        src = self.layers[src_idx]
        trg = self.layers[trg_idx]
        # Ensure that both layers are of the same type before copying
        if type(src) != type(trg):
            raise ValueError("Source and target layers must be of the same type to copy parameters")
        # Copy parameters from src to trg
        trg.load_state_dict(src.state_dict())

    def double_layers(self):
        n = self.n_layers
        for idx in range(n):
            self.prepend_layer()
            self.copy_layer(n, 0)

    def forward(self, input_ids, depth=None):
        """
        Note: adds 1 to length, as it inserts a 0th column for bos token
        """
        if depth is None:
            depth = self.n_layers
        x = self.text_input(input_ids)
        xs = [x]

        if self.random_layers:
            # shuffled_list = list(range(self.n_layers))
            # random.shuffle(shuffled_list)
            # for idx in shuffled_list[:depth]:
            for d in range(depth):
                # print(f'layer {d}')
                idx = random.randint(0, self.n_layers-1)
                layer = self.layers[idx]
                if self.checkpointing:
                    x = checkpoint(layer, x)
                else:
                    x = layer(x)
                xs.append(x)
        else:
            for layer in self.layers[:depth]:
                if self.checkpointing:
                    x = checkpoint(layer, x)
                else:
                    x = layer(x)
                xs.append(x)
        xs = torch.stack(xs, dim=0)
        probs = self.text_output(xs) # (n_layers+1, batch_size, example_length, n_vocab_out)
        return probs
    
    def losses(self, output_ids, depth=None, tracking=0.0):
        input_ids = output_ids[...,:-1]
        probs = self.forward(input_ids, depth=depth) # (n_layers+1, batch_size, example_length, n_vocab_out)
        output_ids = torch.stack([output_ids] * probs.shape[0]).unsqueeze(-1) # (n_layers+1, batch_size, example_length, 1) uint8_t data (torch.long storage)
        cross_entropy = -torch.gather(probs, dim=-1, index=output_ids).squeeze(-1).log2() # cross entropy samples
        if tracking > 0.0:
            p = probs[-1].detach().unsqueeze(0)
            q = probs # model prob
            cross_entropy = cross_entropy - tracking*torch.sum(p * (q.log2() - p.log2()), dim=-1)
        return cross_entropy

    def train(self, batch, lr=1e-8, beta=0.99, mu=1e-7):
        if type(batch) is str:
            batch = bytes(txt[idx:idx+example_length], encoding='utf8')
        if type(batch) is bytes:
            batch = torch.tensor(list(batch)).long().view(1, -1).to('cuda')
        losses = self.losses(batch)
        losses = torch.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0)
        loss = torch.mean(losses)
        loss.backward()        
        for name, p in self.named_parameters():
            p.data.mul_(1.0 - mu * torch.sign(p.grad) * torch.sign(p.data))
            p.data.sub_(torch.sign(p.grad), alpha=lr)
            p.grad.data *= beta
        return losses.detach().cpu().numpy()

    def test(self, batch):
        if type(batch) is str:
            batch = bytes(txt[idx:idx+example_length], encoding='utf8')
        if type(batch) is bytes:
            batch = torch.tensor(list(batch)).long().view(1, -1).to('cuda')
        with torch.no_grad():
            losses = self.losses(batch)
            losses = torch.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0)
            loss = torch.mean(losses)
        return losses.detach().cpu().numpy()

    @torch.no_grad()
    def generate(self, input_ids=None, generation_data=None, output_layer=-1, temp=1.0, pattern=None):
        ## pattern -- if not None, gives the pattern of layers to call in order.
        if input_ids is None:
            device = next(self.parameters()).device
            input_ids = torch.tensor([], torch.long).to(device).view(1,0)

        if pattern is None:
            pattern = list(range(self.n_layers))
        
        if generation_data is None:
            if pattern is None:
                generation_data = [[] for _ in range(self.n_layers)]  # not the same as [[]]*self.n_layers, which gives SAME list.
            else:
                generation_data = [[] for _ in pattern]

        x = self.text_input(input_ids)

        if self.n_layers > 0:
            try:
                n = generation_data[0][0].shape[-2]
            except:
                n = 0
            x = x[...,n:,:]
            xs = [x]
            for layer_idx, layer_data in zip(pattern, generation_data):
                layer = self.layers[layer_idx]
                x = layer(x, layer_data=layer_data)
                xs.append(x)
                #xs = torch.stack(xs, dim=0)
            probs = self.text_output(xs[output_layer].unsqueeze(0))[0,:,-1,:] # (n_layers+1, batch_size, example_length, n_vocab_out)
        else:
            probs = self.text_output(x.unsqueeze(0))[0,:,-1,:] # (n_layers+1, batch_size, example_length, n_vocab_out)

        if temp > 0:
            y = torch.distributions.Categorical(probs=probs**(1.0/temp)).sample().unsqueeze(-1)
        else:
            y = torch.argmax(probs, keepdim=True)
        return torch.cat((input_ids, y), dim=-1), generation_data

    def serialize(self):
        return {'state_dict': self.state_dict(),
                'config': {
                    'n_vocab_in': self.n_vocab_in,
                    'n_vocab_out': self.n_vocab_out,
                    'bos': self.bos,
                    'd_model': self.d_model,
                    'n_layers': self.n_layers,
                    'input_config': self.input_config,
                    'input_classname': self.input_classname,
                    'input_classcode': self.input_classcode,
                    'output_config': self.output_config,
                    'output_classname': self.output_classname,
                    'output_classcode': self.output_classcode,
                    'layer_config': self.layer_config,
                    'layer_classname': self.layer_classname,
                    'layer_classcode': self.layer_classcode}}

    def autocomplete(self,
                     prompt=None,
                     temp=1.0,
                     n_generate=512,
                     n_ctx=None,
                     encoder=utf8encode,
                     decoder=utf8decode,
                     output_layer=-1,
                     pattern=None,
                     burst=64):
        Categorical = torch.distributions.Categorical
        if prompt is None:
            prompt = """In a shocking finding, scientists discovered a herd of unicorns living in a
remote, previously unexplored valley in the Andes Mountains. Even more
surprising to the researchers was the fact that the unicorns spoke perfect
English."""
        device = next(self.parameters()).device
        input_ids = torch.tensor([c for c in encoder(prompt)], dtype=torch.long, device=device).unsqueeze(0)
        if n_ctx is not None:
            input_ids = input_ids[...,-n_ctx:]
        if type(pattern) is int:
            pattern = [random.randint(0, self.n_layers-1) for _ in range(pattern)]
        async def sampler(input_ids, pattern):            
            generation_data = None
            for idx in range(n_generate):
                if burst > 0 and idx % burst == burst-1:
                    await asyncio.sleep(0)  # Give other tasks a chance to run
                input_ids, generation_data = self.generate(input_ids=input_ids, generation_data=generation_data, output_layer=output_layer, temp=temp, pattern=pattern)
                if n_ctx is not None:
                    input_ids = input_ids[...,-n_ctx:]
                    generation_data = [layer_data[-n_ctx:] for layer_data in generation_data] ## todo, why not use a tensor? lists of lists was premature optimization, i think
                yield input_ids

        return sampler(input_ids, pattern)
    
    async def display_autocomplete(self,
                     prompt=None,
                     n_ctx=None,
                     temp=1.0,
                     n_generate=512,
                     encoder=utf8encode,
                     decoder=utf8decode,
                     burst=64,
                     output_layer=-1,
                     pattern=None):
        
        sampler = self.autocomplete(prompt=prompt, temp=temp, n_generate=n_generate, n_ctx=n_ctx,
                                    encoder=encoder, decoder=decoder, output_layer=output_layer, burst=burst, pattern=pattern)
        display_handle = display(HTML(prompt), display_id=True)
        list_of_tokens = []
        cnt = 0
        async for input_ids in sampler:
            c = input_ids.view(-1)[-1]
            list_of_tokens.append(c)
            cnt += 1
            if cnt % 8 == 0:
                continue
            completion = decoder(list_of_tokens)
            n_lines = prompt.count("\n") + completion.count("\n")
            def cleanup(s):
                return s.replace('<', '&lt;').replace('>', '&gt;')
            contents = ('<pre style="background: black; color: lime; font-family: monospace">'+
                    '<span style="color: white">' + cleanup(prompt) + '</span>' + cleanup(completion) + '\n'*max(1, 20-n_lines) + '</pre>')
            display_handle.update(HTML(contents))
        return display_handle


    def save(self, path):
        checkpoint = {
            'model': self.serialize(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        model = LanguageModel(**checkpoint['model']['config'])
        model.load_state_dict(checkpoint['model']['state_dict'])
        return model

    def save_version(self, path, max_versions=4):
        # Get the base path without the extension
        base_path = Path(path).with_suffix('')

        # Get all existing versions
        existing_versions = [
            p for p in base_path.parent.glob(f"{base_path.name}-*") if p.is_file()
        ]

        # If there are max_versions or more, delete the oldest one
        if len(existing_versions) >= max_versions:
            oldest_version = min(existing_versions, key=os.path.getctime)
            os.remove(oldest_version)

        # Find the highest current version number
        if existing_versions:
            max_version = max(int(p.stem.split('-')[-1]) for p in existing_versions)
        else:
            max_version = -1

        # Use the next version number for the new save
        new_version = max_version + 1

        # Create new versioned file name
        new_path = f"{base_path}-{new_version}.pt"
        
        # Save to the new path
        self.save(new_path)
    
    @classmethod
    def load_version(cls, path):
        # Get the base path without the extension
        base_path = Path(path).with_suffix('')

        # Get all existing versions
        existing_versions = [
            p for p in base_path.parent.glob(f"{base_path.name}-*") if p.is_file()
        ]

        # If there are no versions, raise an exception
        if not existing_versions:
            raise ValueError(f"No versions of {path} found")

        # Find the highest current version number
        max_version = max(int(p.stem.split('-')[-1]) for p in existing_versions)

        # Construct path to most recent version
        recent_path = f"{base_path}-{max_version}.pt"

        # Load the mything from the most recent path
        return cls.load(path=recent_path)
