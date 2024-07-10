import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional
from src.modules.misc import get_act_fn
from src.modules.norms import MaskedBatchNorm1d, LayerScale
from src.modules.masked_convs import Conv1dSame
from src.modules import layers
from src.modules.alibi import get_alibi
from functools import partial


class MaskedConv1d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=17,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv = Conv1dSame(in_channels,
                               out_channels,
                               kernel_size,
                               groups=groups,
                               stride=stride,
                               dilation=dilation,
                               bias=bias)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.stride > 1:
                mask = mask[:, ::self.stride]
        return mask

    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(~mask[:, None, :],
                              torch.tensor(0., dtype=x.dtype, device=x.device))
            # mask = self.compute_mask(x, mask)
        x = self.conv(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x, mask=None):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0], ) + (1, ) * (
            x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class CustomGLU(nn.GLU):

    def __init__(self, dim: int, act: str = 'sigmoid'):
        super().__init__()
        if act != 'sigmoid':
            self.dim = dim
            self.activation = get_act_fn(act)()
        else:
            # sigmoid -> use builtin CUDA kernel
            self.activation = None

    def forward(self, inputs: Tensor) -> Tensor:
        if self.activation is None:
            return super().forward(inputs)
        else:
            outputs, gate = inputs.chunk(2, dim=self.dim)
            return outputs * self.activation(gate)


class GLUMlp(nn.Module):

    def __init__(self,
                 dim: int = 512,
                 expand: int = 4,
                 dropout: float = 0.1,
                 bias: bool = True,
                 activation: str = 'gelu') -> None:
        super(GLUMlp, self).__init__()

        self.ffn1 = nn.Linear(dim, dim * expand, bias=bias)
        self.glu = CustomGLU(dim=-1, act=activation)
        self.do1 = nn.Dropout(p=dropout)
        self.ffn2 = nn.Linear(dim * expand // 2, dim, bias=bias)
        # self.do2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.glu(x)
        x = self.do1(x)
        x = self.ffn2(x)
        # x = self.do2(x)
        return x


class AltAttention(nn.Module):

    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim**-0.5
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=True)
        # self.proj_drop = nn.Dropout(dropout)

    def forward(self, inputs, mask=None, alibi_bias=None):
        qkv = self.qkv(inputs)
        qkv = qkv.view(-1, inputs.shape[1], self.num_heads,
                       self.dim * 3 // self.num_heads).permute(0, 2, 1, 3)
        q, k, v = qkv.split([self.dim // self.num_heads] * 3, dim=-1)

        if mask is not None:
            mask = mask[:, None, None, :]

        attn = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale

        if alibi_bias is not None:
            attn = attn.type_as(alibi_bias)
            attn += alibi_bias

        attn = layers.MaskedSoftmax(dim=-1)(attn, mask=mask)  #.to(q.dtype)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.permute(0, 2, 1, 3).reshape(-1, inputs.shape[1], self.dim)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x


class AltBlock(nn.Module):

    def __init__(self,
                 dim=256,
                 num_heads=4,
                 expand=4,
                 attn_dropout=0.2,
                 mlp_dropout=0.2,
                 drop_path=0.,
                 activation='gelu',
                 prenorm=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.norm1 = nn.LayerNorm(
            dim)  #MaskedBatchNorm1d(dim, momentum=0.05, channels_last=True)
        self.self_attn = AltAttention(dim=dim,
                                      num_heads=num_heads,
                                      dropout=attn_dropout)
        self.drop1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(
            dim)  #MaskedBatchNorm1d(dim, momentum=0.05, channels_last=True)
        self.mlp = GLUMlp(dim,
                          expand,
                          dropout=mlp_dropout,
                          activation=activation)
        self.drop2 = DropPath(drop_path)

        self.prenorm = prenorm
        self.attn_scale = LayerScale(dim)
        self.mlp_scale = LayerScale(dim)

    def forward(self, inputs, mask=None, alibi_bias=None):
        x = inputs
        if self.prenorm:
            x = self.norm1(x)
        x = self.self_attn(x, mask=mask, alibi_bias=alibi_bias)
        x = self.drop1(x)
        x = self.attn_scale(x)
        x = x + inputs
        if not self.prenorm:
            x = self.norm1(x)
        attn_out = x

        if self.prenorm:
            x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop2(x)
        x = self.mlp_scale(x)
        x = x + attn_out
        if not self.prenorm:
            x = self.norm2(x)
        return x


class Conv1DBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size=17,
                 groups=4,
                 dilation=1,
                 stride=1,
                 conv_dropout=0.0,
                 mlp_dropout=0.0,
                 drop_path=0.0,
                 expand=4,
                 activation='swish',
                 prenorm=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.prenorm = prenorm
        self.stride = stride

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.glu = CustomGLU(dim=-1, act=activation)
        self.expand_conv = nn.Linear(dim, 2 * dim)
        self.conv = MaskedConv1d(dim,
                                 dim,
                                 kernel_size=kernel_size,
                                 groups=groups)
        self.conv_norm = MaskedBatchNorm1d(dim, momentum=0.05)
        self.conv_act = get_act_fn(activation)()
        self.conv_proj = nn.Linear(dim, dim)
        self.mlp = GLUMlp(dim, expand, mlp_dropout, activation=activation)
        self.conv_dropout = nn.Dropout(conv_dropout)
        # self.mlp_dropout = nn.Dropout(mlp_dropout)
        self.drop1 = DropPath(drop_path)
        self.drop2 = DropPath(drop_path)
        self.conv_scale = LayerScale(dim)
        self.mlp_scale = LayerScale(dim)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.stride > 1:
                mask = mask[:, ::self.stride]
        return mask

    def forward(self, inputs, mask=None):
        x = inputs
        if self.prenorm:
            x = self.norm1(x)
        x = self.expand_conv(x)
        x = self.glu(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x, mask=mask)
        mask = self.compute_mask(inputs, mask)
        x = self.conv_norm(x, mask=mask)
        x = self.conv_act(x)
        x = self.conv_dropout(x)
        x = x.permute(0, 2, 1)
        x = self.conv_proj(x)
        x = self.drop1(x)
        x = self.conv_scale(x)
        if self.stride == 1:
            x = x + inputs
        if not self.prenorm:
            x = self.norm1(x)

        conv_out = x
        if self.prenorm:
            x = self.norm2(x)
        x = self.mlp(x)
        # x = self.mlp_dropout(x)
        x = self.drop2(x)
        x = self.mlp_scale(x)
        if self.stride == 1:
            x = x + conv_out
        if not self.prenorm:
            x = self.norm2(x)
        return x


class SqueezeformerBlock(nn.Module):

    def __init__(self,
                 dim=256,
                 kernel_size=17,
                 groups=4,
                 num_heads=4,
                 conv_expand=4,
                 attn_expand=4,
                 num_conv_block=1,
                 num_attn_block=1,
                 conv_dropout=0.1,
                 attn_dropout=0.1,
                 mlp_dropout=0.1,
                 drop_path=0.1,
                 activation='swish',
                 prenorm=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.conv_blocks = nn.ModuleList([
            Conv1DBlock(dim, kernel_size, groups, 1, 1, conv_dropout,
                        mlp_dropout, drop_path, conv_expand, activation,
                        prenorm) for _ in range(num_conv_block)
        ])
        self.attn_blocks = nn.ModuleList([
            AltBlock(dim, num_heads, attn_expand, attn_dropout, mlp_dropout,
                     drop_path, activation, prenorm)
            for _ in range(num_attn_block)
        ])

    def forward(self, inputs, mask=None, alibi_bias=None):
        x = inputs  #(B,N,C)
        for block in self.conv_blocks:
            x = block(x, mask=mask)
        for block in self.attn_blocks:
            x = block(x, mask=mask, alibi_bias=alibi_bias)
        return x


class SqueezeformerEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_dim, dropout=0.0, embed_proj=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embed_proj:
            self.embed_proj = nn.Linear(embed_dim, embed_dim)
            self.embed_drop = nn.Dropout(dropout)
        else:
            self.embed_proj = None
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        if self.embed_proj is not None:
            x = self.embed_drop(x)
            x = self.embed_proj(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class SqueezefomerEncoder(nn.Module):

    def __init__(self,
                 dim=384,
                 conv_ksize=17,
                 conv_groups=4,
                 num_heads=4,
                 num_layers=12,
                 conv_expand=4,
                 attn_expand=4,
                 conv_num_blocks=1,
                 num_attn_blocks=1,
                 conv_dropout=0.1,
                 attn_dropout=0.1,
                 mlp_dropout=0.1,
                 droppath=0.1,
                 act='swish',
                 prenorm=False,
                 alibi=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.alibi = alibi
        self.layers = nn.ModuleList([
            SqueezeformerBlock(dim, conv_ksize, conv_groups, num_heads,
                               conv_expand, attn_expand, conv_num_blocks,
                               num_attn_blocks, conv_dropout, attn_dropout,
                               mlp_dropout, droppath, act, prenorm)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        if self.alibi:
            alibi_bias = get_alibi(x.size(1),
                                   self.num_heads).to(dtype=x.dtype,
                                                      device=x.device).repeat(
                                                          x.size(0), 1, 1, 1)
        else:
            alibi_bias = None
        outputs = []
        for layer in self.layers:
            x = layer(x, mask=mask, alibi_bias=alibi_bias)
            if mask is not None and hasattr(layer, 'compute_mask'):
                mask = layer.compute_mask(x, mask)
            outputs.append(x)
        return outputs


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


class SqueezeformerModel(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model
        self._head_type = cfg.head.type
        self.embedding = SqueezeformerEmbedding(cfg.vocab_size, cfg.dim,
                                                cfg.embedding.dropout,
                                                cfg.embedding.proj)
        self.encoder = SqueezefomerEncoder(
            cfg.dim,
            conv_groups=cfg.encoder.conv_groups,
            num_heads=cfg.encoder.attn_heads,
            num_layers=cfg.encoder.depth,
            conv_ksize=cfg.encoder.conv_ksize,
            conv_expand=cfg.encoder.conv_expand,
            attn_expand=cfg.encoder.attn_expand,
            conv_num_blocks=cfg.encoder.conv_num_blocks,
            num_attn_blocks=cfg.encoder.attn_num_blocks,
            conv_dropout=cfg.encoder.conv_dropout,
            attn_dropout=cfg.encoder.attn_dropout,
            mlp_dropout=cfg.encoder.mlp_dropout,
            droppath=cfg.encoder.droppath,
            act=cfg.encoder.act,
            prenorm=cfg.encoder.prenorm,
            alibi=cfg.encoder.alibi,
        )
        self.head_dropout = nn.Dropout(cfg.head.dropout)
        if cfg.head.rnn:
            self.rnn = nn.GRU(cfg.dim,
                              cfg.dim,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=cfg.head.bi_rnn)
            out_dim = (2 if cfg.head.bi_rnn else 1) * cfg.dim
        else:
            self.rnn = None
            out_dim = cfg.dim

        self.is_masked_pool = ('masked'
                               in cfg.pool_type) or 'attn' in cfg.pool_type
        self.is_channel_last = ('attn' in cfg.pool_type)
        if 'attn' in cfg.pool_type:
            self.pool = POOLS[cfg.pool_type](cfg.dim)
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

    def forward(self, input_ids, mask):
        x = self.embedding(input_ids)
        x = self.encoder(x, mask)[-1]
        x = self.head_dropout(x)
        if self.rnn is not None:
            x, _h_n = self.rnn(x)
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
