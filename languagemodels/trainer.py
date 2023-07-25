# Author: Shaun Harker
# Date: 2023-07-02
# License: MIT

import torch
import time
import os
from pathlib import Path
from .dataset import FastPileBytesDataset
from .model import LanguageModel
from .optimizer import CustomAdamW
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import IPython
import torch


class Trainer:
    def __init__(self,
                 example_length=512,
                 batch_size=1,
                 model=None,
                 lr=1e-6,
                 betas=(.9, .999),
                 weight_decay=0.001,
                 batch_multiplier=8,
                 prefix=None):
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

        self.dataset = FastPileBytesDataset(prefix=prefix, example_length=example_length)
        self.model = model.to('cuda')
        self.optimizer = CustomAdamW(
            [{'params': layer.parameters()} for layer in self.model.layers] +
            [{'params': self.model.text_input.parameters()}] +
            [{'params': [p]} for p in self.model.text_output.parameters()],
            lr=self.config['lr'], betas=self.config['betas'], weight_decay=self.config['weight_decay'], batch_multiplier=self.config['batch_multiplier'])
        self.inbox = []


    def update_dataset(self, prefix=None, example_length=None):
        self.dataset = FastPileBytesDataset(prefix=prefix, example_length=example_length)

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
    def load(cls, path, prefix=None):
        checkpoint = torch.load(path)
        model = LanguageModel(**checkpoint['model']['config'])
        model.load_state_dict(checkpoint['model']['state_dict'])
        trainer = cls(**checkpoint['config'], model=model, prefix=prefix)
        return trainer
        
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
        return time.time(), batch, losses


def save_version(trainer, path, max_versions=4):
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
    
    # Save the mything to the new path
    trainer.save(new_path)
    

def load_version(path, prefix=None):
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
    return Trainer.load(path=recent_path, prefix=prefix)


def display_layercode(code):
    formatter = HtmlFormatter(style='monokai')
    return IPython.display.HTML('<style type="text/css">{}</style>{}'.format(
        formatter.get_style_defs('.highlight'),
        highlight(code, PythonLexer(), formatter)))
