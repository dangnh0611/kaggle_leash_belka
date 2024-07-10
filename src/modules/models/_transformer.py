from torch import nn
from src.modules import layers
import torch
import logging
from functools import partial
from transformers import AutoConfig, AutoTokenizer, AutoModel, DataCollatorWithPadding
from modules.models._transformer import RobertaModel, RobertaConfig


class FirstPooling(nn.Module):

    def forward(self, x, padding_mask = None):
        return x[:, 0, :]


POOLS = {
    'avg': partial(nn.AdaptiveAvgPool1d, output_size=1),
    'max': partial(nn.AdaptiveMaxPool1d, output_size=1),
    'masked_max': layers.GlobalMaskedMaxPooling1d,
    'masked_mean': layers.GlobalMaskedAvgPooling1d,
    'first': FirstPooling,
}


class TransformerModel(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model

        if cfg.model_path is not None:
            self.encoder = AutoModel.from_pretrained(cfg.model_path,
                                                 add_pooling_layer=False,
                                                 vocab_size=cfg.vocab_size,
                                                 pad_token_id=0,
                                                 bos_token_id=0,
                                                 eos_token_id=0,
                                                 ignore_mismatched_sizes=True)

        # GLOBAL POOL
        assert cfg.pool_type in [
            'avg', 'max', 'masked_max', 'masked_avg', 'first'
        ]
        self.is_masked = ('masked' in cfg.pool_type)
        self.pool = POOLS[cfg.pool_type]()

        # HEAD
        if cfg.head == 'mlp':
            self.head = layers.MLP(in_channels=self.encoder.config.hidden_size,
                                   hidden_channels=cfg.mlp_chans + [3],
                                   norm_layer=None,
                                   act_layer=nn.ReLU,
                                   inplace=False,
                                   bias=True,
                                   dropout=cfg.dropout,
                                   last_norm=False,
                                   last_activation=False,
                                   last_dropout=False)
        else:
            raise ValueError
        
    def forward(self, x, padding_mask):
        x = self.encoder(x, padding_mask.float())[0]
        x = x.permute(0, 2, 1)  # NTC -> NCT
        if self.is_masked:
            x = self.pool(x, padding_mask)
        else:
            x = self.pool(x).squeeze(-1)
        x = self.head(x)
        return x
