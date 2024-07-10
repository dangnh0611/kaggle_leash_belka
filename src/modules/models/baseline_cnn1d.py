from torch import nn
from src.modules import layers
import torch
import logging
from functools import partial
from src.modules.misc import make_net_dims, get_act_fn
from src.modules.norms import get_norm_layer

logger = logging.getLogger(__name__)

POOLS = {
    'avg': partial(nn.AdaptiveAvgPool1d, output_size=1),
    'max': partial(nn.AdaptiveMaxPool1d, output_size=1),
    'masked_max': layers.GlobalMaskedMaxPooling1d,
    'masked_avg': layers.GlobalMaskedAvgPooling1d,
    'masked_gem': partial(layers.GlobalMaskedGEMPooling1d),
    'attn': partial(layers.AttentionPooling1d, channel_first=True),
}


class Model(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model
        self._has_nnfp = cfg.nnfp
        conv_layers = []

        cnn_dims = [cfg.encoder.base_dim] + make_net_dims(
            cfg.encoder.depth - 1,
            cfg.encoder.base_dim,
            cfg.encoder.dim_scale_method,
            width_multiplier=1,
            divisor=8,
        )

        act_layer = get_act_fn(cfg.encoder.act)
        cur_dim = cfg.encoder.embed_dim
        for i, _out_dim in enumerate(cnn_dims):
            norm_layer = cfg.encoder.norm
            if norm_layer is not None:
                if 'instancenorm' in norm_layer or 'ibnorm' in norm_layer:
                    if i >= (len(cnn_dims) // 2):
                        norm_layer = norm_layer.replace(
                            'instancenorm',
                            'batchnorm').replace('ibnorm', 'batchnorm')

            norm_layer = get_norm_layer(
                norm_layer) if norm_layer is not None else None

            if norm_layer is None:
                conv_layers.extend([
                    layers.CausalConv1d(cur_dim,
                                        _out_dim,
                                        kernel_size=cfg.encoder.ksize,
                                        stride=1),
                    act_layer(),
                ])
            else:
                if cfg.encoder.act_before_norm:
                    conv_layers.extend([
                        layers.CausalConv1d(cur_dim,
                                            _out_dim,
                                            kernel_size=cfg.encoder.ksize,
                                            stride=1),
                        act_layer(),
                        norm_layer(_out_dim),
                    ])
                else:
                    conv_layers.extend([
                        layers.CausalConv1d(cur_dim,
                                            _out_dim,
                                            kernel_size=cfg.encoder.ksize,
                                            stride=1),
                        norm_layer(_out_dim),
                        act_layer(),
                    ])

            cur_dim = _out_dim

        if cfg.small_init_embed:
            # Small Embedding init + LN: https://github.com/BlinkDL/SmallInitEmb
            embed_layers = [
                nn.Embedding(cfg.vocab_size,
                             cfg.encoder.embed_dim,
                             padding_idx=None),  # NLC
                nn.LayerNorm(cfg.encoder.embed_dim),
                layers.TransposeLast(),  # NCL
            ]
        else:
            embed_layers = [
                nn.Embedding(cfg.vocab_size,
                             cfg.encoder.embed_dim,
                             padding_idx=None),  # NLC
                layers.TransposeLast(),  # NCL
            ]
        self.layers = nn.Sequential(*embed_layers, *conv_layers)

        # GLOBAL POOL
        # assert cfg.pool_type in [
        #     'avg', 'max', 'masked_max', 'masked_avg', 'masked_gem'
        # ]
        self.is_masked = ('masked' in cfg.pool_type) or cfg.pool_type == 'attn'
        if cfg.pool_type == 'attn':
            self.pool = POOLS[cfg.pool_type](cur_dim)
        else:
            self.pool = POOLS[cfg.pool_type]()

        # Neural Fingerprint
        if self._has_nnfp:
            self.nnfp = nn.Sequential(
                nn.Linear(cur_dim, cfg.nnfp_dim, bias=True),
                nn.BatchNorm1d(cfg.nnfp_dim), nn.GELU(),
                nn.Linear(cfg.nnfp_dim, cfg.nnfp_dim, bias=True),
                nn.BatchNorm1d(cfg.nnfp_dim), nn.Sigmoid())
            cur_dim = cfg.nnfp_dim

        # HEAD
        self.head = layers.MLP(in_channels=cur_dim,
                               hidden_channels=cfg.head.mlp_chans +
                               [cfg.head.num_output],
                               norm_layer=get_norm_layer(cfg.head.norm)
                               if cfg.head.norm is not None else None,
                               act_layer=get_act_fn(cfg.head.act),
                               inplace=False,
                               bias=True,
                               dropout=cfg.head.dropout,
                               last_norm=False,
                               last_activation=False,
                               last_dropout=False)

        if cfg.weight_init == 'tf':
            logger.info('Apply TF weight initialisation..')
            from src.utils.misc import tf_init_weights
            self.apply(tf_init_weights)
        elif cfg.weight_init == 'torch':
            pass
        else:
            raise ValueError

        if cfg.small_init_embed:
            from src.utils.misc import small_init_embed
            self.apply(small_init_embed)

    def forward(self, x, mask):
        x = self.layers(x)
        if self.is_masked:
            x = self.pool(x, mask)
        else:
            x = self.pool(x).squeeze(-1)
        if self._has_nnfp:
            x = self.nnfp(x)
            x = (x > 0.5).float()
        x = self.head(x)
        return x
