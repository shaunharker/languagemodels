# Author: Shaun Harker
# Date: 2023-07-02
# License: MIT

import torch
import time
import asyncio
import numpy as np
from IPython.display import display, HTML
import re
import collections
import string

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

        self.clear_stats()


    def update_lr(self, lr, layer_idx=None):
        for idx, group in enumerate(self.optimizer.param_groups):
            if (layer_idx is None) or (idx == layer_idx):
                group['lr'] = lr
    
    def insert_layer(self, index, **optim_args):
        params = self.model.insert_layer(index=index)
        self.optimizer.add_param_group({'params': params, **optim_args})

    def append_layer(self, **optim_args):
        self.insert_layer(index=self.model.n_layers)

    def prepend_layer(self, **optim_args):
        self.insert_layer(index=0)

    def save(self, path):
        checkpoint = {
            'model': self.model.serialize(),
            'config': self.config,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        model = LanguageModel(**checkpoint['model']['config'])
        model.load_state_dict(checkpoint['model']['state_dict'])
        trainer = cls(**checkpoint['config'], model=model)
        return trainer

    def clear_stats(self):
        self.inbox = []
        self.batches = []
        self.losses = []
        self.times = []
        self.n = 0
        self.D = 0 # total data
        
    def batch(self, batch_size=None, example_length=None):
        if batch_size is None:
            batch_size = self.config['batch_size']
        if example_length is None:
            example_length = self.config['example_length']        
        if len(self.inbox) > 0:
            batch = self.inbox.pop()
        else:
            batch = self.dataset.batch(batch_size=batch_size, example_length=example_length)
        return batch
            
    def train(self, depth=None):
        # read config
        if depth is None:
            depth = self.model.n_layers
        batch_size = self.config['batch_size']
        example_length = self.config['example_length']

        # get batch
        batch = self.batch(batch_size, example_length)
        
        # perform forward pass

        losses = self.model.losses(batch, depth=depth)
        losses = torch.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0)
        loss = torch.mean(losses)

        # perform optimization
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # compute stats
        batch = batch.cpu().numpy()
        losses = losses.detach().cpu().numpy()
        self.batches.append(batch)
        self.losses.append(losses)
        self.times.append(time.time())

        self.n += 1
        self.D += batch_size * example_length
