from torch import nn
from torch import nn
from src.modules import layers
import torch
import logging
from functools import partial
from src.modules.misc import get_act_fn
from torch.nn import functional as F
import numpy as np

NORMS = {'LN': nn.LayerNorm, 'BN': nn.BatchNorm1d}


class SineLayer(nn.Module):
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 is_first=False,
                 omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class IFMMLPModel(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model

        layers = []
        out_dim = cfg.in_dim
        for i, dim in enumerate(cfg.mlp_chans):
            layers.append(
                SineLayer(out_dim,
                          dim,
                          is_first=(i == 0),
                          omega_0=cfg.first_omega_0
                          if i == 0 else cfg.hidden_omega_0))
            layers.append(get_act_fn(cfg.act)())
            layers.append(nn.Dropout(cfg.dropout))
            out_dim = dim
        self.head = nn.Linear(out_dim, 3)
        with torch.no_grad():
            self.head.weight.uniform_(
                -np.sqrt(6 / out_dim) / cfg.hidden_omega_0,
                np.sqrt(6 / out_dim) / cfg.hidden_omega_0)

        self.layers = nn.Sequential(*layers)

    def forward(self, x, padding_mask=None):
        x = self.layers(x.float())
        x = self.head(x)
        return x
