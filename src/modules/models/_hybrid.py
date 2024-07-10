import math
from collections import OrderedDict
from dataclasses import dataclass, replace, field
from functools import partial
from typing import Callable, Optional, Union, Tuple, List
import torch
from torch import nn
from torch.jit import Final
from modules.models.squeezeformer import SqueezeformerBlock
from src.modules.misc import (
    get_act_fn,
    get_norm_layer,
    make_divisible,
    make_net_dims,
)


def make_block(block_name, cfg, dim, stride, droppath):
    if block_name == 'squeeze':
        return SqueezeformerBlock.from_config(cfg, dim, stride, droppath)
    else:
        raise ValueError


class HybridEncoder(nn.Module):

    def __init__(self, cfg):
        # make net dim
        net_dims = [cfg.base_dim] + make_net_dims(
            cfg.depth - 1,
            cfg.base_dim,
            cfg.dim_scale_method,
            width_multiplier=cfg.width_multiplier,
            divisor=getattr(cfg, 'divisor', 8),
        )
        



    def _init_from_topo(self, cfg, topo, droppath = 0.0, droppath_mode = 'linear'):
        """
        Config is a list of [block, dim, stride].
        For example:
        [
            [squeeze, 128, 2],
            [squeeze, 256, 1]
        ]
        """
        super().__init__()
        num_blocks = len(topo)
        
        if droppath_mode == 'linear':
            droppaths = torch.linspace(0, droppath, num_blocks)
        elif droppath_mode == 'constant':
            droppaths = [droppath] * num_blocks
        else:
            raise ValueError
        
        self.blocks = nn.ModuleList()
        for i, (block_name, dim, stride) in enumerate(topo):
            dpr = droppaths[i]
            block = make_block(block_name, cfg, dim, stride, dpr)
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
