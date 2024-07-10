from torch import nn
from torch import nn
from src.modules import layers
import torch
import logging
from functools import partial
from transformers import AutoConfig, AutoTokenizer, AutoModel, DataCollatorWithPadding
from transformers import RobertaModel, RobertaConfig
from src.modules.misc import get_act_fn

NORMS = {'LN': nn.LayerNorm, 'BN': nn.BatchNorm1d}


class MlpModel(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model
        self.mlp = layers.MLP(
            cfg.in_dim,
            cfg.mlp_chans + [3],
            norm_layer=NORMS[cfg.norm] if cfg.norm is not None else None,
            act_layer=get_act_fn(cfg.act),
            dropout=cfg.dropout,
            last_norm=False,
            last_activation=False,
            last_dropout=False)

    def forward(self, x, padding_mask=None):
        x = self.mlp(x.float())
        return x
