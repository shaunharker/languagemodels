# Author: Shaun Harker
# Date: 2023-07-02
# License: MIT

import torch
from torch.nn import Module, Linear, Embedding, ModuleList, GELU, LayerNorm
from torch.nn.functional import pad, softmax, one_hot
from .dataset import utf8decode, utf8encode
import inspect
from IPython.display import display, HTML
import asyncio
from torch.utils.checkpoint import checkpoint
import random
from pathlib import Path

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


class TextInputEmbedding(Module):
    def __init__(self, n_vocab_in, d_model, bos=0, init_scale=1.0):
        super().__init__()
        self.bos = bos
        self.embedding = Embedding(n_vocab_in, d_model)
        self.embedding.weight.data *= init_scale

    def forward(self, input_ids):
        padded_input_ids = pad(input_ids, (1, 0), "constant", self.bos)
        x = self.embedding(padded_input_ids)
        return x

    def insert_layer(self, index, **kwargs):
        return []
    
    def append_layer(self, **kwargs):
        return []

    def prepend_layer(self, **kwargs):
        return []


class TextInputMultipleEmbedding(Module):
    def __init__(self, n_vocab_in, d_embd, d_model, bos=0, init_scale=1.0):
        super().__init__()
        self.bos = bos
        self.embedding = Embedding(n_vocab_in, d_embd)
        self.embedding.weight.data *= init_scale
        self.lookback = d_model // d_embd 

    def forward(self, input_ids):
        padded_input_ids = pad(input_ids, (1, 0), "constant", self.bos)
        xs = []
        xs.append(self.embedding(padded_input_ids))
        for _ in range(self.lookback-1):
            xs.append(torch.roll(xs[-1], shifts=1, dims=-2))
            xs[-1][...,0,:] = xs[-1][...,1,:] # copy bos embedding
        return torch.cat(xs, dim=-1)
    
    def insert_layer(self, index, **kwargs):
        return []
    
    def append_layer(self, **kwargs):
        return []

    def prepend_layer(self, **kwargs):
        return []

    
class TextInputAutoregressive(Module):
    def __init__(self, n_vocab_in, d_model, bos=0, bias=False, init_scale=0.0):
        super().__init__()
        self.bos = bos
        self.embedding = Embedding(n_vocab_in, d_model)
        self.M = Linear(d_model, d_model, bias=bias)
        self.M.weight.data *= init_scale

    def forward(self, input_ids):
        padded_input_ids = pad(input_ids, (1, 0), "constant", self.bos)
        x = self.embedding(padded_input_ids)
        ys = [x[...,0,:]]
        for idx in range(1, x.shape[-2]):
            ys += [self.M(ys[-1]) + x[...,idx,:]]
        return torch.stack(ys, dim=-2)

    def insert_layer(self, index, **kwargs):
        return []
    
    def append_layer(self, **kwargs):
        return []

    def prepend_layer(self, **kwargs):
        return []


class TextOutputReadHeads(Module):
    def __init__(self, n_vocab_out, d_model, n_layers, bias=True, init_scale=1.0):
        super().__init__()
        self.n_vocab_out = n_vocab_out
        self.d_model = d_model
        self.bias = bias
        self.read_heads = ModuleList(Linear(d_model, n_vocab_out, bias=self.bias) for _ in range(n_layers+1))
        for read_head in self.read_heads:
            read_head.weight.data *= init_scale
            if self.bias:
                read_head.bias.data *= init_scale

    def forward(self, x):
        # x.shape == (n_layers+1, batch_size, example_length, n_vocab_out)
        logits = torch.stack([self.read_heads[idx](x[idx]) for idx in range(x.shape[0])])
        probs = softmax(logits, dim=-1)
        return probs
    
    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except:
            return 'cpu'
        
    def insert_layer(self, index, **kwargs):
        # note: off by 1 due to embedding layer, treat read_heads as if starting from 1 (though copy embedding layer read head)
        layer = Linear(self.d_model, self.n_vocab_out, bias=self.bias).to(self.device)
        layer.weight.data.copy_(self.read_heads[index].weight.data)
        if self.bias:
            layer.bias.data.copy_(self.read_heads[index].bias.data)
        self.read_heads.insert(index=index+1, module=layer)
        return list(self.read_heads[index+1].parameters())
    
    def append_layer(self, **kwargs):
        n_layers = len(self.read_heads)-1
        return self.insert_layer(index=n_layers)
    
    def prepend_layer(self, **kwargs):
        return self.insert_layer(index=0)


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
