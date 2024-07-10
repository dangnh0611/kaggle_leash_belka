from torch import nn
from src.modules import layers
import torch
import logging
from functools import partial
from transformers import AutoConfig, AutoTokenizer, AutoModel, DataCollatorWithPadding
from transformers import DebertaV2Config, DebertaV2Model
from src.modules.misc import get_act_fn


POOLS = {
    'avg': partial(nn.AdaptiveAvgPool1d, output_size=1),
    'max': partial(nn.AdaptiveMaxPool1d, output_size=1),
    'masked_max': layers.GlobalMaskedMaxPooling1d,
    'masked_avg': layers.GlobalMaskedAvgPooling1d,
    'masked_gem': partial(layers.GlobalMaskedGEMPooling1d),
    'attn': partial(layers.AttentionPooling1d, channel_first=False),
    'concat_attn': partial(layers.ConcatAttentionPooling1d,
                           channel_first=False),
}


class DebertaModel(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model
        self._head_type = cfg.head.type

        debertav2_config = DebertaV2Config(**cfg.encoder)
        self.encoder = DebertaV2Model(debertav2_config)
        out_dim = cfg.encoder.hidden_size

        self.is_masked_pool = ('masked'
                               in cfg.pool_type) or 'attn' in cfg.pool_type
        self.is_channel_last = ('attn' in cfg.pool_type)
        if 'attn' in cfg.pool_type:
            self.pool = POOLS[cfg.pool_type](cfg.encoder.hidden_size)
        else:
            self.pool = POOLS[cfg.pool_type]()

        head_dim = out_dim if cfg.pool_type != 'concat_attn' else 2 * out_dim

        if self._head_type in ['leash', 'mtr']:
            self.head = layers.MLP(head_dim,
                                   cfg.head.mlp_chans + [cfg.head.num_output],
                                   norm_layer=cfg.head.norm,
                                   act_layer=get_act_fn(cfg.head.act),
                                   dropout=cfg.head.dropout,
                                   last_norm=False,
                                   last_activation=False,
                                   last_dropout=False)
        elif self._head_type == 'mtr_mlm':
            # No dropout for regression
            self.mtr_head = layers.MLP(head_dim,
                                       cfg.head.mlp_chans + [189],
                                       norm_layer=cfg.head.norm,
                                       act_layer=get_act_fn(cfg.head.act),
                                       dropout=0.0,
                                       last_norm=False,
                                       last_activation=False,
                                       last_dropout=False)
            self.mlm_head = layers.MLP(out_dim,
                                       cfg.head.mlp_chans + [cfg.vocab_size],
                                       norm_layer=cfg.head.norm,
                                       act_layer=get_act_fn(cfg.head.act),
                                       dropout=cfg.head.dropout,
                                       last_norm=False,
                                       last_activation=False,
                                       last_dropout=False)
        else:
            raise ValueError

    def forward(self, x, mask):
        x = self.encoder(x, mask.float())[0]
        if not self.is_channel_last:
            x = x.permute(0, 2, 1)  # NLC -> NCL
        if self.is_masked_pool:
            pooled_output = self.pool(x, mask)
        else:
            pooled_output = self.pool(x).squeeze(-1)

        if self._head_type != 'mtr_mlm':
            logits = self.head(pooled_output)
            return logits
        else:
            mtr_logits = self.mtr_head(pooled_output)
            mlm_logits = self.mlm_head(x)
            return mtr_logits, mlm_logits
