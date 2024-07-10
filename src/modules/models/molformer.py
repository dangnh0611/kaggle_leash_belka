from torch import nn
from src.modules import layers
import torch
import logging
from functools import partial
from transformers import AutoConfig, AutoTokenizer, AutoModel, DataCollatorWithPadding
from transformers import RobertaModel, RobertaConfig
from src.modules.misc import get_act_fn
import logging


logger = logging.getLogger(__name__)


class FirstPooling(nn.Module):

    def forward(self, x, padding_mask=None):
        return x[:, 0, :]


POOLS = {
    'avg': partial(nn.AdaptiveAvgPool1d, output_size=1),
    'max': partial(nn.AdaptiveMaxPool1d, output_size=1),
    'masked_max': layers.GlobalMaskedMaxPooling1d,
    'masked_avg': layers.GlobalMaskedAvgPooling1d,
    'first': FirstPooling,
    'masked_gem': partial(layers.GlobalMaskedGEMPooling1d)
}


class MolFormer(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model

        self.encoder = AutoModel.from_pretrained(cfg.model_path,
                                                 trust_remote_code=True,
                                                 add_pooling_layer=False,
                                                 ignore_mismatched_sizes=True,
                                                 **cfg.encoder)
        
        # Frozen some layers
        frozen_layers = []
        assert cfg.num_trainable_layers >= 0
        # self.encoder.encoder.layer is a ModuleList
        if cfg.num_trainable_layers > len(self.encoder.encoder.layer):
            # all layers will be trained
            pass
        elif cfg.num_trainable_layers == 0:
            logger.warn('Frozen all layers, including Embedding layers..')
            frozen_layers.append(self.encoder)
        else:
            frozen_layers.extend(self.encoder.encoder.layer[:-cfg.num_frozen_layers])
        for layer in frozen_layers:
            for param in layer.parameters():
                param.requires_grad = False

        # GLOBAL POOL
        assert cfg.pool_type in [
            'avg', 'max', 'masked_max', 'masked_avg', 'first'
        ]
        self.is_masked = ('masked' in cfg.pool_type)
        self.pool = POOLS[cfg.pool_type]()

        # HEAD
        self.head = layers.MLP(self.encoder.config.hidden_size,
                               cfg.head.mlp_chans + [3],
                               norm_layer=cfg.head.norm,
                               act_layer=get_act_fn(cfg.head.act),
                               dropout=cfg.head.dropout,
                               last_norm=False,
                               last_activation=False,
                               last_dropout=False)

    def forward(self, x, padding_mask):
        x = self.encoder(x, padding_mask.float())[0]
        # if x.isnan().any():
        #     print('Nan in features')
        x = x.permute(0, 2, 1)  # NTC -> NCT
        if self.is_masked:
            x = self.pool(x, padding_mask)
        else:
            x = self.pool(x).squeeze(-1)
        x = self.head(x)
        return x
