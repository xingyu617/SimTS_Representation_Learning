import random
from typing import Union, Callable, Optional, List

import numpy as np
from einops import reduce, rearrange
import torch
from torch import nn
import torch.nn.functional as F

from .dilation import *


class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    @param chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size ==0:
          return x
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)

class WeightNorm(torch.nn.Module):
    append_g = '_g'
    append_v = '_v'

    def __init__(self, module, weights=['weight']):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            g = torch.norm(w)
            v = w/g.expand_as(w)
            g = nn.Parameter(g.data)
            v = nn.Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v

            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)

    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v*(g/torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()
        
        # Computes left padding so that the applied convolutions are causal
        self.padding = (kernel_size - 1) * dilation
        padding = self.padding
        # First causal convolution
        self.conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(0.1)

        # Second causal convolution
        self.conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(0.1)

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.ReLU() if final else None
        
    def forward(self, x):
       
        out_causal=self.conv1(x)
        out_causal=self.chomp1(out_causal)
        out_causal=self.dropout1(F.gelu(out_causal))
        out_causal=self.conv2(out_causal)
        out_causal=self.chomp2(out_causal)
        out_causal=self.dropout2(F.gelu(out_causal))
        res = x if self.upordownsample is None else self.upordownsample(x)
        
        if self.relu is None:
            x = out_causal + res
            
        else:
            x= self.relu(out_causal + res)
        
        return x


def generate_binomial_mask(x,p=0.5):
  
    mask = torch.from_numpy(np.random.binomial(1, p, size=(x.size(0), x.size(1)))).to(torch.bool).to(x.device)
    #nan_mask = ~x.isnan().any(axis=-1)
    #x[~nan_mask] = 0
    #mask &= nan_mask
    x[~mask] = 0
    return x


class CausalCNNEncoder(torch.nn.Module): 
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).

    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, 
                 in_channels,
                 reduced_size,
                 component_dims, 
                 kernel_list=[1,2, 4, 8, 16, 32, 64, 128],
                 ):
        super(CausalCNNEncoder, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = 'cpu'
        self.input_fc = CausalConvolutionBlock(in_channels, reduced_size, 1, 1)
        self.repr_dropout = torch.nn.Dropout(p=0.1)
        self.kernel_list = kernel_list
        self.multi_cnn = nn.ModuleList(
            [nn.Conv1d(reduced_size, component_dims, k, padding=k-1) for k in kernel_list]
        )

    def print_para(self):
        
        return list(self.multi_cnn.parameters())[0].clone()    
        
    def forward(self, x_h, x_f = None, mask = None, train = True):

        nan_mask_h = ~x_h.isnan().any(axis=-1)
        x_h[~nan_mask_h] = 0
        
        x_h = x_h.transpose(2,1)
        x_h = self.input_fc(x_h)
        x_h = x_h.transpose(2,1)
        
        if mask =='binomial':
            mask = generate_binomial_mask(x_h.size(0), x_h.size(1)).to(x_h.device)
            mask &= nan_mask_h
            x_h[~mask] = 0
        
        x_h = x_h.transpose(2,1)
        
        if train:

            nan_mask_f = ~x_f.isnan().any(axis=-1)
            x_f[~nan_mask_f] = 0
            x_f = x_f.transpose(2,1)
            x_f = self.input_fc(x_f)
            x_f = x_f.transpose(2,1)
            x_f = x_f.transpose(2,1)

            trend_h = []
            trend_h_weights = []
            trend_f = []
            trend_f_weights = []

            for idx, mod in enumerate(self.multi_cnn):
              
                out_h = mod(x_h)  # b d t
                out_f = mod(x_f)
                if self.kernel_list[idx] != 1:
                  out_h = out_h[..., :-(self.kernel_list[idx] - 1)]
                  out_f = out_f[..., :-(self.kernel_list[idx] - 1)]

                trend_h.append(out_h.transpose(1,2))  # b 1 t d
                trend_h_weights.append(out_h.transpose(1, 2)[:,-1,:].unsqueeze(-1)) 

                trend_f.append(out_f.transpose(1,2))  # b 1 t d
                trend_f_weights.append(out_f.transpose(1, 2)[:,-1,:].unsqueeze(-1)) 

            trend_h = reduce(
              rearrange(trend_h, 'list b t d -> list b t d'),
              'list b t d -> b t d', 'mean'
            )

            trend_f = reduce(
              rearrange(trend_f, 'list b t d -> list b t d'),
              'list b t d -> b t d', 'mean'
            )

            trend_h = self.repr_dropout(trend_h)
            trend_h_repr = trend_h[:,-1,:]
            # Select random column
            # rand_idx = random.randint(0, trend_h.size(1)-1)
            # trend_h_repr = trend_h[:,rand_idx,:]

            return trend_h, trend_h_repr, trend_f.detach()

        else:
            trend_h = []
            trend_h_weights = []

            for idx, mod in enumerate(self.multi_cnn):
              
                out_h = mod(x_h)  # b d t

                if self.kernel_list[idx] != 1:
                  out_h = out_h[..., :-(self.kernel_list[idx] - 1)]
                trend_h.append(out_h.transpose(1,2))  # b 1 t d
                trend_h_weights.append(out_h.transpose(1, 2)[:,-1,:].unsqueeze(-1)) 

            trend_h = reduce(
              rearrange(trend_h, 'list b t d -> list b t d'),
              'list b t d -> b t d', 'mean'
            )

            trend_h = self.repr_dropout(trend_h)
            trend_h_repr = trend_h[:,-1,:]

            return trend_h, trend_h_repr, None