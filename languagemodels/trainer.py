# Author: Shaun Harker
# Date: 2023-07-02
# License: MIT

import torch
import time
import asyncio
import numpy as np
from IPython.display import display, HTML

from .dataset import FastPileBytesDataset
from .model import LanguageModel
from .transformer import Transformer
from .optimizer import CustomAdamW

class Trainer:
    def __init__(self,
                 example_length=512,
                 batch_size=1,
                 n_vocab_in=256,
                 n_vocab_out=256,
                 module=None,
                 bos=0,
                 lr=1e-6,
                 betas=(.9, .999),
                 weight_decay=0.001,
                 batch_multiplier=8):
        """
        example_length: length of examples to use for training
        batch_size: number of examples in a single batch
        batch_multiplier: number of batches to accumulate gradients on
                          before calling optimizer.step. This makes for
                          an effective batch_size of batch_size*batch_multiplier,
                          hence the name.
        """
        self.config = {
            'example_length': example_length,
            'batch_size': batch_size,
            'n_vocab_in': n_vocab_in,
            'n_vocab_out': n_vocab_out,
            'bos': bos,
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'batch_multiplier': batch_multiplier
        }

        self.dataset = FastPileBytesDataset(example_length=512)
        if module is not None:
            self.install_module(module)

        self.inbox = []
        self.loss_by_layer = []
        self.last_layer_loss = []
        self.times = []
        self.n = 0
        self.D = 0 # total data

    def install_module(self, module):
        self.module = module
        self.model = LanguageModel(n_vocab_in=self.config['n_vocab_in'],
                                   n_vocab_out=self.config['n_vocab_out'],
                                   bos=self.config['bos'],
                                   module=module).to('cuda')
        self.optimizer = CustomAdamW(
            [{'params': layer.parameters()} for layer in self.model.module.layers] +
            [{'params': self.model.text_input.parameters()}] +
            [{'params': self.model.text_output.parameters()}],
            lr=self.config['lr'], betas=self.config['betas'], weight_decay=self.config['weight_decay'], batch_multiplier=self.config['batch_multiplier'])

    def update_lr(self, lr, layer_idx=None):
        for idx, group in enumerate(self.optimizer.param_groups):
            if (layer_idx is None) or (idx == layer_idx):
                group['lr'] = lr
    
    def save(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            #'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'loss_by_layer': self.loss_by_layer,
            'last_layer_loss': self.last_layer_loss,
            'times': self.times,
            'n': self.n,
            'D': self.D
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, module=None):
        checkpoint = torch.load(path)
        config = checkpoint['config']
        trainer = cls(**config)  # Make Trainer instance with same arguments
        if module is None:
            module = Transformer()
        trainer.install_module(module)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        #trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.loss_by_layer = checkpoint['loss_by_layer']
        trainer.last_layer_loss = checkpoint['last_layer_loss']
        trainer.times = checkpoint['times']
        trainer.n = checkpoint['n']
        trainer.D = checkpoint['D']
        trainer.time_repair()
        return trainer

    def clear_stats(self):
        self.inbox = []
        self.loss_by_layer = []
        self.last_layer_loss = []
        self.times = []
        self.n = 0
        self.D = 0 # total data

    def time_repair(self, gap=10.0):
        """
        Aligns time-stamps with present (bringings into future) by
        detecting and removing large time gaps. By default, a gap
        of 10 seconds is assumed to be do to halted training, and
        thus will be removed and replaced with an average time gap.
        """
        excess = self.times[0]
        sum_dt = 0.00
        cnt_dt = 0.01
        for idx in range(len(self.times)-1):
            dt = self.times[idx+1] - self.times[idx]
            self.times[idx] -= excess
            if dt > gap:
                excess += dt - sum_dt/cnt_dt
            else:
                sum_dt += dt
                cnt_dt += 1
        self.times[-1] -= excess
        delta = time.time() - self.times[-1] 
        self.times = [t + delta for t in self.times]

    def train(self):
        # read config
        batch_size = self.config['batch_size']
        example_length = self.config['example_length']

        # get batch of training examples
        if len(self.inbox) > 0:
            batch = self.inbox.pop()
        else:
            batch = self.dataset.batch(batch_size=batch_size,
                                       example_length=example_length)
        
        # perform forward pass
        losses = self.model.losses(batch)
        losses = torch.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0)
        loss = torch.mean(losses)

        # perform optimization
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # compute stats
        batch_loss_by_layer = np.mean(losses.detach().cpu().numpy(), axis=(1,2)) # over bs and pos
        batch_last_layer_loss_by_pos = np.mean(losses[-1].detach().cpu().numpy(), axis=0) # over bs
        batch_last_layer_loss = np.mean(batch_last_layer_loss_by_pos)
        
        # save data
        self.loss_by_layer.append(batch_loss_by_layer)
        self.last_layer_loss.append(batch_last_layer_loss)
        self.times.append(time.time())
        self.n += 1
        self.D += batch_size * example_length

    def autocomplete(self,
                     prompt=None,
                     temp=1.0,
                     n_generate=512,
                     n_ctx=None,
                     output_layer=-1):
        Categorical = torch.distributions.Categorical
        if prompt is None:
            prompt = """In a shocking finding, scientists discovered a herd of unicorns living in a
remote, previously unexplored valley in the Andes Mountains. Even more
surprising to the researchers was the fact that the unicorns spoke perfect
English."""
        n_vocab_out = self.config['n_vocab_out']
        input_ids = self.dataset.encode(prompt)
        if n_ctx is None:
            n_ctx = 512
        input_ids = input_ids[-n_ctx:]
        prompt = self.dataset.decode(input_ids)
        device='cuda'
        async def sampler(input_ids):
            for _ in range(n_generate):
                await asyncio.sleep(0)  # Give other tasks a chance to run
                probs = self.model(torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0))[output_layer].view(-1)[-n_vocab_out:]
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
                     output_layer=-1):
        
        sampler = self.autocomplete(prompt=prompt, temp=temp, n_generate=n_generate, n_ctx=n_ctx, output_layer=output_layer)
        display_handle = display(HTML(prompt), display_id=True)
        list_of_tokens = []
        async for c in sampler:
            list_of_tokens.append(c)
            completion = self.dataset.decode(list_of_tokens)
            def cleanup(s):
                return s.replace('<', '&lt;').replace('>', '&gt;')
            contents = ('<pre style="background: black; color: lime; font-family: monospace">'+
                    '<span style="color: white">' + cleanup(prompt) + '</span>' + cleanup(completion) + '\n'*20 + '</pre>')
            display_handle.update(HTML(contents))
        return display_handle
