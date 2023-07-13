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


class MeanAndDriftEstimator:
    def __init__(self, decay_rates=[.99, .999]):
        self.decay_rates = np.array(decay_rates)
        self.loss_emas = np.zeros_like(decay_rates)
        self.time_emas = np.zeros_like(decay_rates)
        self.n = 0

    def __call__(self, loss, time):
        self.loss_emas = self.decay_rates * self.loss_emas + (1 - self.decay_rates) * loss
        self.time_emas = self.decay_rates * self.time_emas + (1 - self.decay_rates) * time
        self.n += 1

    def get_stats(self):
        delta_loss = self.loss_emas[1] - self.loss_emas[0]
        delta_time = self.time_emas[1] - self.time_emas[0]
        rate_of_change = delta_loss / delta_time if delta_time != 0 else 0
        return self.n, self.loss_emas, self.time_emas, rate_of_change
    

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

        punctuation = string.punctuation
        escaped_punctuation = [re.escape(p) for p in punctuation]
        punctuation_regex = b"|".join(p.encode() for p in escaped_punctuation)
        self.pattern = re.compile(rb'\w+|\W')
        self.word_loss_dict = collections.defaultdict(lambda: (0, 0))
        self.word_emas = collections.defaultdict(lambda: MeanAndDriftEstimator())

    def update_lr(self, lr, layer_idx=None):
        for idx, group in enumerate(self.optimizer.param_groups):
            if (layer_idx is None) or (idx == layer_idx):
                group['lr'] = lr
    
    def insert_layer(self, index, **optim_args):
        params = self.model.insert_layer(index=index)
        self.optimizer.add_param_group({'params': params, **optim_args})
        self.loss_by_layer = [np.concatenate((x[:index], [x[index], x[index]], x[index+1:])) for x in self.loss_by_layer]

    def append_layer(self, **optim_args):
        self.insert_layer(index=self.model.n_layers)

    def prepend_layer(self, **optim_args):
        self.insert_layer(index=0)

    def save(self, path):
        checkpoint = {
            'model': self.model.serialize(),
            'config': self.config,
            'loss_by_layer': self.loss_by_layer,
            'last_layer_loss': self.last_layer_loss,
            'times': self.times,
            'n': self.n,
            'D': self.D,
            'word_loss_dict': list(self.word_loss_dict.items())
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        model = LanguageModel(**checkpoint['model']['config'])
        model.load_state_dict(checkpoint['model']['state_dict'])
        trainer = cls(**checkpoint['config'], model=model)
        trainer.loss_by_layer = checkpoint['loss_by_layer']
        trainer.last_layer_loss = checkpoint['last_layer_loss']
        trainer.times = checkpoint['times']
        trainer.n = checkpoint['n']
        trainer.D = checkpoint['D']
        try:
            for (k,v) in checkpoint['word_loss_dict']:
                trainer.word_loss_dict[k] = v
        except:
            pass
        trainer.time_repair()
        return trainer

    def clear_stats(self):
        self.inbox = []
        self.loss_by_layer = []
        self.last_layer_loss = []
        self.times = []
        self.n = 0
        self.D = 0 # total data

    def condense_data(self):

        # courtesy GPT-4 with a bit of prodding:
        def avg_every_other(arr, axis):
            # Determine size of the axis
            size = arr.shape[axis]
            pair_num = size // 2

            # Split the array into main part and possible leftover
            main_part = np.swapaxes(arr, axis, -1)[..., :pair_num*2]
            leftover = np.swapaxes(arr, axis, -1)[..., pair_num*2:]

            # Reshape the main part for averaging
            reshaped = main_part.reshape(*main_part.shape[:-1], -1, 2)

            # Calculate the average and concatenate the leftover if exists
            averaged = reshaped.mean(axis=-1)
            if size % 2 == 1:  # if size along the axis is odd
                result = np.concatenate([averaged, leftover], axis=-1)
            else:
                result = averaged

            return np.swapaxes(result, -1, axis)
        
        def condense(list_of_arr):
            avged = avg_every_other(np.stack(list_of_arr), axis=0)
            return [np.squeeze(x, axis=0) for x in np.split(avged, len(avged))]

        self.loss_by_layer = condense(self.loss_by_layer)
        self.last_layer_loss = condense(self.last_layer_loss)
        self.times = condense(self.times)

        
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
            
    def train(self):
        # read config
        batch_size = self.config['batch_size']
        example_length = self.config['example_length']

        # get batch
        batch = self.batch(batch_size, example_length)
        
        # perform forward pass
        losses = self.model.losses(batch)
        losses = torch.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0)
        loss = torch.mean(losses)

        # perform optimization
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # compute stats
        batch = batch.cpu().numpy()
        losses = losses.detach().cpu().numpy()
        batch_loss_by_layer = np.mean(losses, axis=(1,2)) # over bs and pos
        batch_last_layer_loss_by_pos = np.mean(losses[-1], axis=0) # over bs
        batch_last_layer_loss = np.mean(batch_last_layer_loss_by_pos)

        T = time.time()
        for idx in range(losses.shape[1]):
            example_losses = losses[-1,idx,:]
            example = bytes(batch[idx].tolist())
            matches = self.pattern.finditer(example)
            for match in matches:
                start, end = match.span()
                element = match.group()
                # snazzier way of doing the same we did before with the word_loss_dict
                n, v = self.word_loss_dict[element]
                self.word_loss_dict[element] = (n+1, v+example_losses[start:end])
                self.word_emas[element](sum(example_losses[start:end]), T)
        # save data
        self.loss_by_layer.append(batch_loss_by_layer)
        self.last_layer_loss.append(batch_last_layer_loss)
        self.times.append(T)
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
