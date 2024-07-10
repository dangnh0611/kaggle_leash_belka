from torch import nn
from src.modules import layers
import torch
import logging
from functools import partial
from src.modules.misc import make_net_dims, get_act_fn, make_divisible
from src.modules.norms import get_norm_layer
from torch.nn import init

logger = logging.getLogger(__name__)

POOLS = {
    'avg': partial(nn.AdaptiveAvgPool1d, output_size=1),
    'max': partial(nn.AdaptiveMaxPool1d, output_size=1),
    'masked_max': layers.GlobalMaskedMaxPooling1d,
    'masked_avg': layers.GlobalMaskedAvgPooling1d,
    'masked_gem': partial(layers.GlobalMaskedGEMPooling1d),
    'attn': partial(layers.AttentionPooling1d, channel_first=True),
}


# https://github.com/amedprof/Feedback-Prize--English-Language-Learning/blob/main/src/model_zoo/pooling.py
class MaskedMeanPooling(nn.Module):

    def __init__(self):
        super(MaskedMeanPooling, self).__init__()

    def forward(self, hidden_state, padding_mask):
        """
        hidden_state: NCT
        padding_mask: NT (bool, True is not pad, False is pad)
        """
        padding_mask_expanded = padding_mask.unsqueeze(1).expand(
            hidden_state.size())  # NCT
        sum_embeddings = torch.sum(hidden_state * padding_mask_expanded,
                                   -1)  # NCT -> NC
        # @TODO (dangnh): optimize
        sum_mask = padding_mask_expanded.sum(-1)  # NCT -> NC
        # sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask  # NC
        return mean_embeddings


class SEAttention(nn.Module):

    def __init__(self,
                 inp_dim,
                 hidden_dim,
                 hidden_act=nn.ReLU,
                 scale_act=nn.Sigmoid):
        super().__init__()
        self.mean_pool = MaskedMeanPooling()
        self.fc = nn.Sequential(nn.Linear(inp_dim, hidden_dim, bias=False),
                                hidden_act(inplace=True),
                                nn.Linear(hidden_dim, inp_dim, bias=False),
                                scale_act())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, padding_mask=None):
        """
        x: NCT
        padding_mask: NT
        """
        y = self.mean_pool(x, padding_mask)  # NC
        y = self.fc(y).unsqueeze(-1)  # NC -> NC -> NC1
        return x * y.expand_as(x)  # NCT


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.mean_pool = MaskedMeanPooling()
        self.conv = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2)
        self.act = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, padding_mask=None):
        """
        x: NCT
        padding_mask: NT
        """
        # NCT
        y = self.mean_pool(x, padding_mask).unsqueeze(1)  # NCT -> NC -> N1C
        y = self.act(self.conv(y)).permute(0, 2, 1)  # N1C -> N1C -> NC1
        return x * y.expand_as(x)  # NCT


class ConvBlock(nn.Module):

    def __init__(self, in_dim, out_dim, ksize, act, norm, attn, dropout=0.0):
        super().__init__()
        norm_layer = get_norm_layer(norm) if norm is not None else None
        self.conv = layers.CausalConv1d(in_dim,
                                        out_dim,
                                        kernel_size=ksize,
                                        stride=1)
        self.act = get_act_fn(act)()
        self.norm = norm_layer(
            out_dim) if norm_layer is not None else nn.Identity()
        if attn == 'SE':
            se_hidden_dim = make_divisible(out_dim // 4, 8)
            # mobilenetv3 use HardSigmoid instead of Sigmoid
            # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py#L52C35-L52C35
            self.attn = SEAttention(out_dim,
                                    se_hidden_dim,
                                    hidden_act=nn.ReLU,
                                    scale_act=nn.Sigmoid)
        elif attn == 'ECA':
            self.attn = ECAAttention(kernel_size=5)
        elif attn is None:
            self.attn = None
        else:
            raise ValueError(f'Invalid channel attention {attn}')
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        # residual = x
        x = self.norm(self.act(self.conv(x)))
        x = self.attn(x, mask)
        x = self.drop(x)
        # x = x + residual
        return x


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
        self.embed_layers = nn.Sequential(*embed_layers)

        self.blocks = nn.ModuleList()
        cur_dim = cfg.encoder.embed_dim
        for i, _out_dim in enumerate(cnn_dims):
            block = ConvBlock(cur_dim, _out_dim, cfg.encoder.ksize,
                              cfg.encoder.act, cfg.encoder.norm,
                              cfg.encoder.attn, cfg.encoder.dropout)
            self.blocks.append(block)
            cur_dim = _out_dim

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
        x = self.embed_layers(x)

        for block in self.blocks:
            x = block(x, mask)

        if self.is_masked:
            x = self.pool(x, mask)
        else:
            x = self.pool(x).squeeze(-1)
        if self._has_nnfp:
            x = self.nnfp(x)
            x = (x > 0.5).float()
        x = self.head(x)
        return x
