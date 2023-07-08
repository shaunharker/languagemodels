# Author: Shaun Harker
# Date: 2023-07-02
# License: MIT

import torch
from torch.nn import Module, Linear, Embedding, ModuleList
from torch.nn.functional import pad, softmax, one_hot
from .dataset import utf8decode, utf8encode
import inspect
from IPython.display import display, HTML
import asyncio
from torch.utils.checkpoint import checkpoint

class TextInput(Module):
    def __init__(self, n_vocab_in, d_model, bos=0, **kwargs):
        super().__init__()
        self.bos = bos
        self.n_vocab_in = n_vocab_in
        self.d_model = d_model
        assert n_vocab_in <= d_model

    def forward(self, input_ids):
        return one_hot(pad(input_ids, (1, 0), "constant", self.bos), self.d_model).float()

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

    def append_layer(self, **kwargs):
        return []

    def prepend_layer(self, **kwargs):
        return []
    

class TextOutputReadHeads(Module):
    def __init__(self, n_vocab_out, d_model, n_layers, bias=True, init_scale=0.0):
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
        
    def append_layer(self, **kwargs):
        layer = Linear(self.d_model, self.n_vocab_out, bias=self.bias).to(self.device)
        layer.weight.data.copy_(self.read_heads[-1].weight.data)
        if self.bias:
            layer.bias.data.copy_(self.read_heads[-1].bias.data)
        self.read_heads.append(layer)
        return list(self.read_heads[-1].parameters())

    def prepend_layer(self, **kwargs):
        layer = Linear(self.d_model, self.n_vocab_out, bias=self.bias).to(self.device)
        layer.weight.data.copy_(self.read_heads[0].weight.data)
        if self.bias:
            layer.bias.data.copy_(self.read_heads[0].bias.data)
        self.read_heads.insert(0, layer)
        return list(self.read_heads[0].parameters())


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
                 checkpointing=True):
        super().__init__()
        self.n_vocab_in = n_vocab_in
        self.n_vocab_out = n_vocab_out
        self.bos = bos
        self.d_model = d_model
        self.n_layers = n_layers
        self.checkpointing = checkpointing

        def load_class(code, classname):
            local_vars = {}
            exec(code, {'torch': torch,
                        'Module': Module,
                        'ModuleList': ModuleList,
                        'Linear': Linear,
                        'Embedding': Embedding,
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

    def append_layer(self):
        self.layers.append(self.layer_class(**self.layer_config).to(self.device))
        params = list(self.layers[-1].parameters())
        params += self.text_input.append_layer(**self.input_config)
        params += self.text_output.append_layer(**self.output_config)
        self.n_layers += 1
        return params
    
    def prepend_layer(self):
        self.layers.insert(0, self.layer_class(**self.layer_config).to(self.device))
        params = list(self.layers[0].parameters())
        params += self.text_input.prepend_layer(**self.input_config)
        params += self.text_output.prepend_layer(**self.output_config)
        self.n_layers += 1
        return params

    def forward(self, input_ids):
        """
        Note: adds 1 to length, as it inserts a 0th column for bos token
        """
        x = self.text_input(input_ids)
        xs = [x]
        for layer in self.layers:
            if self.checkpointing:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
            xs.append(x)
        xs = torch.stack(xs, dim=0)
        probs = self.text_output(xs) # (n_layers+1, batch_size, example_length, n_vocab_out)
        return probs
    
    def losses(self, output_ids):
        input_ids = output_ids[...,:-1]
        probs = self.forward(input_ids) # (n_layers+1, batch_size, example_length, n_vocab_out)
        output_ids = torch.stack([output_ids] * probs.shape[0]).unsqueeze(-1) # (n_layers+1, batch_size, example_length, 1) uint8_t data (torch.long storage)
        return -torch.gather(probs, dim=-1, index=output_ids).squeeze(-1).log2() # cross entropy samples

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
                     encoder = utf8encode,
                     decoder = utf8decode,
                     output_layer=-1):
        Categorical = torch.distributions.Categorical
        if prompt is None:
            prompt = """In a shocking finding, scientists discovered a herd of unicorns living in a
remote, previously unexplored valley in the Andes Mountains. Even more
surprising to the researchers was the fact that the unicorns spoke perfect
English."""
        input_ids = encoder(prompt)
        if n_ctx is None:
            n_ctx = 512
        input_ids = input_ids[-n_ctx:]
        prompt = decoder(input_ids)
        device = next(self.parameters()).device
        async def sampler(input_ids):
            for _ in range(n_generate):
                await asyncio.sleep(0)  # Give other tasks a chance to run
                probs = self(torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0))[output_layer].view(-1)[-self.n_vocab_out:]
                if temp > 0:
                    y = Categorical(probs=probs**(1.0/temp)).sample().item()
                else:
                    y = torch.argmax(probs).item()
                input_ids = (input_ids + [y])[-n_ctx:]
                yield y

        return sampler(list(input_ids))
    
    async def display_autocomplete(self,
                     prompt=None,
                     n_ctx=None,
                     temp=1.0,
                     n_generate=512,
                     encoder = utf8encode,
                     decoder = utf8decode,
                     output_layer=-1):
        
        sampler = self.autocomplete(prompt=prompt, temp=temp, n_generate=n_generate, n_ctx=n_ctx,
                                    encoder=encoder, decoder=decoder, output_layer=output_layer)
        display_handle = display(HTML(prompt), display_id=True)
        list_of_tokens = []
        async for c in sampler:
            list_of_tokens.append(c)
            completion = decoder(list_of_tokens)
            def cleanup(s):
                return s.replace('<', '&lt;').replace('>', '&gt;')
            contents = ('<pre style="background: black; color: lime; font-family: monospace">'+
                    '<span style="color: white">' + cleanup(prompt) + '</span>' + cleanup(completion) + '\n'*20 + '</pre>')
            display_handle.update(HTML(contents))
        return display_handle
