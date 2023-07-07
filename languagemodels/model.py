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

class TextInput(Module):
    def __init__(self, n_vocab_in, d_model, bos=0, **kwargs):
        super().__init__()
        self.bos = bos
        self.n_vocab_in = n_vocab_in
        self.d_model = d_model
        assert n_vocab_in <= d_model

    def forward(self, input_ids):
        return one_hot(pad(input_ids, (1, 0), "constant", self.bos), self.d_model).float()


class TextOutput(Module):
    def __init__(self, n_vocab_out, d_model, **kwargs):
        super().__init__()
        self.n_vocab_out = n_vocab_out
        self.d_model = d_model

    def forward(self, x):
        logits = x[..., -self.n_vocab_out:]
        probs = softmax(logits, dim=-1)
        return probs
    

class TextInputEmbedding(Module):
    def __init__(self, n_vocab_in, d_model, bos=0, **kwargs):
        super().__init__()
        self.bos = bos
        self.embedding = Embedding(n_vocab_in, d_model)

    def forward(self, input_ids):
        padded_input_ids = pad(input_ids, (1, 0), "constant", self.bos)
        x = self.embedding(padded_input_ids)
        return x


class TextOutputReadHeads(Module):
    def __init__(self, n_vocab_out, d_model, n_layers, **kwargs):
        super().__init__()
        self.n_vocab_out = n_vocab_out
        self.d_model = d_model
        self.read_heads = ModuleList(Linear(d_model, n_vocab_out, bias=False) for _ in range(n_layers+1))

    def forward(self, x):
        # x.shape == (n_layers+1, batch_size, example_length, n_vocab_out)
        logits = torch.stack([self.read_heads[idx](x[idx]) for idx in range(x.shape[0])])
        probs = softmax(logits, dim=-1)
        return probs
    
    def add_layer(self, device=None):
        if device is None:
            try:
                device = next(self.parameters()).device
            except:
                device = 'cpu'
        self.read_heads.append(Linear(self.d_model, self.n_vocab_out, bias=False).to(device)) 


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
                 state_dict=None):
        super().__init__()
        self.n_vocab_in = n_vocab_in
        self.n_vocab_out = n_vocab_out
        self.bos = bos
        self.d_model = d_model
        self.n_layers = n_layers
        self.layer_class = layer_class
        self.layer_config = layer_config

        def load_class(code, classname):
            local_vars = {}
            exec(code, {'torch': torch, 'Module': Module, 'softmax': softmax, 'Linear': Linear}, local_vars)
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
            self.input_classcode = inspect.getsource(self.input_class)
        except:
            self.input_classcode = input_classcode

        ## default input config
        if input_config is None:
            input_config = {'n_vocab_in': n_vocab_in, 'bos': bos, 'd_model': d_model}

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
            self.output_classcode = inspect.getsource(self.output_class)
        except:
            self.output_classcode = output_classcode

        ## default output config
        if output_config is None:
            output_config = {'n_vocab_out': n_vocab_out, 'd_model': d_model, 'n_layers': n_layers}

        # layer_class
        if layer_class is None:
            layer_class = load_class(layer_classcode, layer_classname)

        try:
            self.layer_classname = layer_class.__name__
        except:
            self.layer_classname = layer_classname
        
        try:
            self.layer_classcode = inspect.getsource(self.layer_class)
        except:
            self.layer_classcode = layer_classcode
            
        self.text_input = input_class(**input_config)
        self.layers = ModuleList(layer_class(**layer_config) for _ in range(self.n_layers))
        self.text_output = output_class(**output_config)

        if state_dict is not None:
            self.load_state_dict(state_dict)
        

    def add_layer(self, device=None):
        if device is None:
            try:
                device = next(self.parameters()).device
            except:
                device = 'cpu'
        self.layers.append(self.layer_class(**self.layer_config).to(device))
        self.n_layers += 1

    def forward(self, input_ids):
        """
        Note: adds 1 to length, as it inserts a 0th column for bos token
        """
        x = self.text_input(input_ids)
        xs = [x]
        for layer in self.layers:
            x = layer(x)
            xs.append(x)
        xs = torch.stack(xs, dim=0)
        probs = self.text_output(xs) # (n_layers+1, batch_size, example_length, n_vocab_out)
        return probs
    
    def losses(self, output_ids):
        input_ids = output_ids[...,:-1]
        probs = self.forward(input_ids) # (n_layers+1, batch_size, example_length, n_vocab_out)
        output_ids = torch.stack([output_ids] * probs.shape[0]).unsqueeze(-1) # (n_layers+1, batch_size, example_length, 1) uint8_t data (torch.long storage)
        return -torch.gather(probs, dim=-1, index=output_ids).log2() # cross entropy samples

    def serialize(self):
        return {'state_dict': self.state_dict(),
                'n_vocab_in': self.n_vocab_in,
                'n_vocab_out': self.n_vocab_out,
                'bos': self.bos,
                'd_model': self.d_model,
                'n_layers': self.n_layers,
                'input_classname': self.input_classname,
                'input_classcode': self.input_classcode,
                'output_classname': self.output_classname,
                'output_classcode': self.output_classcode,
                'layer_config': self.layer_config,
                'layer_classname': self.layer_classname,
                'layer_classcode': self.layer_classcode}

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
