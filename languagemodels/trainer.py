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
from .optimizer import CustomAdamW

import torch
from torch.nn import Module, Linear, LayerNorm, GELU
from torch.nn.functional import softmax

class Trainer:
    def __init__(self,
                 example_length=512,
                 batch_size=1,
                 model=None,
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
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'batch_multiplier': batch_multiplier
        }

        self.dataset = FastPileBytesDataset(example_length=example_length)
        self.model = model.to('cuda')
        self.optimizer = CustomAdamW(
            [{'params': layer.parameters()} for layer in self.model.layers] +
            [{'params': self.model.text_input.parameters()}] +
            [{'params': [p]} for p in self.model.text_output.parameters()],
            lr=self.config['lr'], betas=self.config['betas'], weight_decay=self.config['weight_decay'], batch_multiplier=self.config['batch_multiplier'])

        self.inbox = []
        self.loss_by_layer = []
        self.last_layer_loss = []
        self.times = []
        self.n = 0
        self.D = 0 # total data

    def update_lr(self, lr, layer_idx=None):
        for idx, group in enumerate(self.optimizer.param_groups):
            if (layer_idx is None) or (idx == layer_idx):
                group['lr'] = lr
    
    def add_layer(self, **optim_args):
        self.model.add_layer()
        self.optimizer.add_param_group({'params': [list(self.model.text_output.parameters())[-1]], **optim_args})

    def save(self, path):
        checkpoint = {
            'model': self.model.serialize(),
            'config': self.config,
            'loss_by_layer': self.loss_by_layer,
            'last_layer_loss': self.last_layer_loss,
            'times': self.times,
            'n': self.n,
            'D': self.D
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        model = LanguageModel(**checkpoint['model'])
        trainer = cls(**checkpoint['config'], model=model)
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
        if len(self.times) == 0:
            return
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

    # def experimental_train(self):
    #     fork = lambda x: x.detach().clone().requires_grad_(True)
    #     layer_pairs = list(zip(self.model.module.layers[:-1], self.model.module.layers[1:]))
    #     if not hasattr(self, 'optimizers'):
    #         optimizers = [optimizer([*layer1.parameters(), *layer2.parameters()]) for (layer1, layer2) in layer_pairs]
    #     x, y = training_pair()
    #     x = x.detach()
    #     for ((layer, next_layer), optimizer) in zip(layer_pairs, optimizers):
    #         optimizer.zero_grad()
    #         criterion(next_layer(layer(fork(x))),y).backward()
    #         optimizer.step()
    #         x = layer(x)
