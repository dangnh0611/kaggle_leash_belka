#Several SqueezeFormer components where copied/ adapted from https://github.com/upskyy/Squeezeformer/

import json
import math
import typing
from typing import Optional, Tuple, Union

import numpy as np
import timm
import torch
from timm.layers.norm_act import BatchNormAct2d
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import (LlamaConfig,
                                                      LlamaRotaryEmbedding)
from transformers.models.speech_to_text import (
    Speech2TextConfig, Speech2TextForConditionalGeneration)
from transformers.models.speech_to_text.modeling_speech_to_text import (
    Speech2TextDecoder, shift_tokens_right)
from src.modules.layers import Downsample1d
from src.modules.norms import LayerScale
from src.modules.masked_convs import MaskedConv1d
from src.modules.misc import get_act_fn
from src.modules.norms import get_norm_layer
from src.modules import layers
from torchvision.ops import StochasticDepth
from src.modules.alibi import get_alibi
from functools import partial


class FFN(nn.Module):

    def __init__(
        self,
        dim: int = 512,
        expand: int = 4,
        act_layer=nn.SiLU,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ffn1 = nn.Linear(dim, int(dim * expand), bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(p=dropout)
        self.ffn2 = nn.Linear(int(dim * expand), dim, bias=True)
        self.drop2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.ffn2(x)
        x = self.drop2(x)
        return x


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


class GLUFFN(nn.Module):

    def __init__(self,
                 dim: int = 512,
                 expand: int = 4,
                 dropout: float = 0.1,
                 glu_act: str = 'gelu') -> None:
        super().__init__()

        self.ffn1 = nn.Linear(dim, dim * expand)
        self.glu = CustomGLU(dim=-1, act=glu_act)
        self.drop1 = nn.Dropout(p=dropout)
        self.ffn2 = nn.Linear(dim * expand // 2, dim)
        self.drop2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.glu(x)
        x = self.drop1(x)
        x = self.ffn2(x)
        x = self.drop2(x)
        return x


class ConvBlock(nn.Module):

    def __init__(
        self,
        dim_in,
        dim_out,
        stride=1,
        kernel_size=17,
        conv_expand=1,
        conv_type='causal',
        conv_norm='masked_batchnorm_1d_first',
        conv_act='silu',
        conv_depthwise=True,
        ffn_expand=2,
        ffn_act='silu',
        dropout=0.0,
        droppath=0.0,
        prenorm=True,
    ):
        super().__init__()
        self._conv_depthwise = conv_depthwise
        self._prenorm = prenorm
        self._stride = stride
        self._conv_type = conv_type

        if stride == 1:
            self.shortcut = nn.Identity()
        elif stride == 2:
            self.shortcut = Downsample1d(dim_in, dim_out, pool_type='conv')
        else:
            raise NotImplementedError

        # # the 'if prenorm' is just to make print(model) look reasonable
        # self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)

        if prenorm:
            self.conv_ln = nn.LayerNorm(dim_in)
        self.expand_conv = nn.Linear(dim_in, dim_in * conv_expand * 2)
        self.glu = CustomGLU(dim=-1, act=ffn_act)

        if conv_type == 'same':
            self.conv = layers.SameConv1d(dim_in * conv_expand,
                                          dim_out,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          groups=dim_in *
                                          conv_expand if conv_depthwise else 1)
        elif conv_type == 'causal':
            self.conv = layers.CausalConv1d(
                dim_in * conv_expand,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                groups=dim_in * conv_expand if conv_depthwise else 1)
        elif conv_type == 'masked':
            self.conv = MaskedConv1d(dim_in * conv_expand,
                                     dim_out,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     groups=dim_in *
                                     conv_expand if conv_depthwise else 1)
        else:
            raise ValueError

        self.conv_norm = get_norm_layer(conv_norm)(dim_out, momentum=0.05)
        self.conv_act = get_act_fn(conv_act)()
        self.conv_dropout = nn.Dropout(dropout)
        if conv_depthwise:
            self.conv_proj = nn.Linear(dim_out, dim_out)
            self.conv_proj_scale = LayerScale(dim_out)
            self.conv_proj_dropout = nn.Dropout(dropout)
        self.conv_proj_droppath = StochasticDepth(p=droppath, mode="row")
        if not prenorm:
            self.conv_ln = nn.LayerNorm(dim_out)
        if prenorm:
            self.mlp_ln = nn.LayerNorm(dim_out)
        self.mlp = GLUFFN(dim_out,
                          expand=ffn_expand,
                          dropout=dropout,
                          glu_act=ffn_act)
        self.mlp_scale = LayerScale(dim_out)
        self.mlp_droppath = StochasticDepth(p=droppath, mode='row')
        if not prenorm:
            self.mlp_ln = nn.LayerNorm(dim_out)

    def compute_mask(self, x, mask=None):
        if mask is not None:
            if self._stride > 1:
                mask = mask[:, ::self._stride]
        return mask

    def forward(self, x, mask=None):
        shortcut = self.shortcut(x)
        if self._prenorm:
            x = self.conv_ln(x)
        x = self.expand_conv(x)
        x = self.glu(x)
        x = x.permute(0, 2, 1)
        if self._conv_type == 'masked':
            x, mask = self.conv(x, mask=mask)
        else:
            x = self.conv(x)
            mask = self.compute_mask(x, mask)
        x = self.conv_norm(x, mask=mask)
        x = self.conv_act(x)
        x = self.conv_dropout(x)
        x = x.permute(0, 2, 1)
        if self._conv_depthwise:
            x = self.conv_proj(x)
            x = self.conv_proj_scale(x)
            x = self.conv_proj_dropout(x)
        x = self.conv_proj_droppath(x)
        if self._stride == 1:
            x = x + shortcut
        if not self._prenorm:
            x = self.conv_ln(x)

        conv_out = x
        if self._prenorm:
            x = self.mlp_ln(x)
        x = self.mlp(x)
        x = self.mlp_scale(x)
        x = self.mlp_droppath(x)
        if self._stride == 1:
            x = x + conv_out
        if not self._prenorm:
            x = self.mlp_ln(x)
        return x, mask


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.squeeze(1)
    k_embed = k_embed.squeeze(1)
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        dim_in=256,
        dim_out=256,
        num_attn_heads=8,
        attn_dropout=0.0,
        pos_embed_type='rope',
        max_position=1,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_attn_heads = num_attn_heads
        self.head_dim = self.dim_in // self.num_attn_heads
        self.scale = self.head_dim**-0.5
        self.pos_embed_type = pos_embed_type
        self.max_position = max_position

        if (self.head_dim * self.num_attn_heads) != self.dim_in:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.dim_in}"
                f" and `num_heads`: {self.num_attn_heads}).")
        self.q_proj = nn.Linear(self.dim_in,
                                self.num_attn_heads * self.head_dim,
                                bias=False)
        self.k_proj = nn.Linear(self.dim_in,
                                self.num_attn_heads * self.head_dim,
                                bias=False)
        self.v_proj = nn.Linear(self.dim_in,
                                self.num_attn_heads * self.head_dim,
                                bias=False)
        if self.pos_embed_type == 'rope':
            self.rotary_embed: LlamaRotaryEmbedding = None
        else:
            self.pos_embed_type = ''
        self.attn_drop = nn.Dropout(attn_dropout)
        self.o_proj = nn.Linear(self.num_attn_heads * self.head_dim,
                                self.dim_out,
                                bias=False)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_attn_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sin=None,
        cos=None,
        alibi_bias=None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_attn_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_attn_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_attn_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if self.pos_embed_type == 'rope':
            if sin is None or cos is None:
                if self.rotary_embed is None:
                    self.rotary_embed = LlamaRotaryEmbedding(
                        self.dim_in // self.num_attn_heads,
                        max_position_embeddings=self.max_position)
                cos, sin = self.rotary_embed(query_states, q_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin)
        else:
            # no positional embedding at all
            pass

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(
            2, 3)) * self.scale

        if attn_weights.size() != (bsz, self.num_attn_heads, q_len,
                                   kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_attn_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}")

        if alibi_bias is not None:
            # attn[:, : alibi_bias.size(1)] += alibi_bias
            attn_weights += alibi_bias

        if attention_mask is not None:
            # if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            #     raise ValueError(
            #         f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            #     )

            # NL -> Nx1x1xL
            attention_mask = attention_mask[:, None, None, :]
            attn_weights.masked_fill_(~attention_mask,
                                      torch.finfo(attn_weights.dtype).min)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
                                                 query_states.dtype)
        attn_weights_dropped = self.attn_drop(attn_weights)
        attn_output = torch.matmul(attn_weights_dropped, value_states)

        if attn_output.size() != (bsz, self.num_attn_heads, q_len,
                                  self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_attn_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.dim_in)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class AltAttention(nn.Module):

    def __init__(self, dim_in=256, dim_out=256, num_heads=4, dropout=0):
        super().__init__()
        self.in_dim = dim_in
        self.out_dim = dim_out
        self.scale = self.in_dim**-0.5
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim_in, 3 * dim_out, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim_out, dim_out, bias=True)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None, alibi_bias=None):
        qkv = self.qkv(hidden_states)
        qkv = qkv.view(-1, hidden_states.shape[1], self.num_heads,
                       self.out_dim * 3 // self.num_heads).permute(0, 2, 1, 3)
        q, k, v = qkv.split([self.out_dim // self.num_heads] * 3, dim=-1)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]

        attn = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale

        if alibi_bias is not None:
            attn = attn.type_as(alibi_bias)
            attn += alibi_bias

        attn = layers.MaskedSoftmax(dim=-1)(attn,
                                            mask=attention_mask)  #.to(q.dtype)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.permute(0, 2, 1, 3).reshape(-1, hidden_states.shape[1],
                                          self.out_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttnBlock(nn.Module):

    def __init__(
        self,
        dim_in=256,
        dim_out=256,
        stride=1,
        attn_type='llama',
        attn_num_heads=4,
        attn_dropout=0.1,
        ffn_expand=4,
        ffn_dropout=0.1,
        ffn_act='gelu',
        droppath=0.0,
        norm='layernorm_1d_last',
        pos_embed_type='rope',
        max_position=1,
        prenorm=True,
    ):
        super().__init__()
        self._prenorm = prenorm

        if stride == 1:
            self.downsample = nn.Identity()
        elif stride == 2:
            self.downsample = Downsample1d(dim_in, dim_out, pool_type='avg2')
        else:
            raise NotImplementedError

        if self._prenorm:
            self.norm1 = get_norm_layer(norm)(dim_out)
        if attn_type == 'llama':
            self.self_attn = LlamaAttention(dim_in=dim_in,
                                            dim_out=dim_out,
                                            num_attn_heads=attn_num_heads,
                                            attn_dropout=attn_dropout,
                                            pos_embed_type=pos_embed_type,
                                            max_position=max_position)
        elif attn_type == 'alt':
            self.self_attn = AltAttention(dim_in=dim_in,
                                          dim_out=dim_out,
                                          num_heads=attn_num_heads,
                                          dropout=attn_dropout)
        else:
            raise ValueError
        self.attn_scale = LayerScale(dim_out)
        self.attn_droppath = StochasticDepth(droppath, mode='row')
        if not self._prenorm:
            self.norm1 = get_norm_layer(norm)(dim_out)
        if self._prenorm:
            self.norm2 = get_norm_layer(norm)(dim_out)
        self.mlp = GLUFFN(
            dim_out,
            ffn_expand,
            dropout=ffn_dropout,
            glu_act=ffn_act,
        )
        self.mlp_scale = LayerScale(dim_out)
        self.mlp_droppath = StochasticDepth(droppath, mode='row')
        if not self._prenorm:
            self.norm2 = get_norm_layer(norm)(dim_out)

    def forward(self,
                hidden_states,
                attention_mask=None,
                sin=None,
                cos=None,
                alibi_bias=None):
        residual = self.downsample(hidden_states)
        x = residual
        if self._prenorm:
            x = self.norm1(x)
        x = self.self_attn(x,
                           attention_mask,
                           sin=sin,
                           cos=cos,
                           alibi_bias=alibi_bias)[0]
        x = self.attn_scale(x)
        x = self.attn_droppath(x)
        x = residual + x
        if not self._prenorm:
            x = self.norm1(x)

        attn_out = x
        if self._prenorm:
            x = self.norm2(x)
        x = self.mlp(x)
        x = self.mlp_scale(x)
        x = self.mlp_droppath(x)
        x = attn_out + x
        if not self._prenorm:
            x = self.norm2(x)
        return x, attention_mask


class SqueezeformerBlock(nn.Module):

    def __init__(
        self,
        stride=1,
        layout='CT',
        dim_in=256,
        dim_out=256,
        conv_ksize=17,
        conv_expand=1,
        conv_depthwise=True,
        conv_act='silu',
        conv_type='causal',
        conv_norm='masked_batchnorm_1d_first',
        attn_type='llama',
        attn_num_heads=4,
        attn_dropout=0.1,
        attn_norm='layernorm_1d_last',
        attn_pos_embed_type='rope',
        ffn_expand=2,
        ffn_dropout=0.1,
        ffn_act='gelu',
        droppath=0.0,
        max_position=1,
        prenorm=True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.block_types = []

        for block_type in layout:
            if block_type == 'C':
                block = ConvBlock(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    stride=stride,
                    kernel_size=conv_ksize,
                    conv_expand=conv_expand,
                    conv_type=conv_type,
                    conv_norm=conv_norm,
                    conv_act=conv_act,
                    conv_depthwise=conv_depthwise,
                    ffn_expand=ffn_expand,
                    ffn_act=conv_act,
                    dropout=ffn_dropout,
                    droppath=droppath,
                    prenorm=prenorm,
                )
                block_type = 'C'
            elif block_type == 'T':
                block = AttnBlock(dim_in=dim_in,
                                  dim_out=dim_out,
                                  stride=stride,
                                  attn_type=attn_type,
                                  attn_num_heads=attn_num_heads,
                                  attn_dropout=attn_dropout,
                                  ffn_expand=ffn_expand,
                                  ffn_dropout=ffn_dropout,
                                  ffn_act=ffn_act,
                                  droppath=droppath,
                                  norm=attn_norm,
                                  pos_embed_type=attn_pos_embed_type,
                                  max_position=max_position,
                                  prenorm=prenorm)
                block_type = 'T'
            else:
                raise ValueError
            self.blocks.append(block)
            self.block_types.append(block_type)

    def forward(self,
                hidden_states,
                attention_mask=None,
                sin=None,
                cos=None,
                alibi_bias=None):
        """
        Args:
            hidden_states: NLC
        """
        for block_type, block in zip(self.block_types, self.blocks):
            if block_type == 'C':
                hidden_states, attention_mask = block(hidden_states,
                                                      attention_mask)
            elif block_type == 'T':
                hidden_states, attention_mask = block(hidden_states,
                                                      attention_mask,
                                                      sin=sin,
                                                      cos=cos,
                                                      alibi_bias=alibi_bias)
        return hidden_states, attention_mask

    @classmethod
    def from_config(cls, cfg, stride, dim_in, dim_out, droppath=0.0):
        kwargs = dict(cfg)
        kwargs.update({
            'stride': stride,
            'dim_in': dim_in,
            'dim_out': dim_out,
            'droppath': droppath
        })
        return cls(**kwargs)


class SqueezeformerEncoder(nn.Module):

    def __init__(self,
                 depth=6,
                 layout='CT',
                 dim_in=256,
                 dim_out=256,
                 conv_ksize=17,
                 conv_expand=1,
                 conv_depthwise=True,
                 conv_act='silu',
                 conv_type='causal',
                 conv_norm='masked_batchnorm_1d_first',
                 attn_type='llama',
                 attn_num_heads=4,
                 attn_dropout=0.1,
                 attn_norm='layernorm_1d_last',
                 attn_pos_embed_type='rope',
                 ffn_expand=2,
                 ffn_dropout=0.1,
                 ffn_act='gelu',
                 droppath=0.0,
                 max_position=1,
                 prenorm=True,
                 droppath_mode='linear'):
        super().__init__()
        assert dim_in == dim_out

        self.attn_num_heads = attn_num_heads
        self.alibi = 'alibi' in attn_pos_embed_type
        self.rope = 'rope' in attn_pos_embed_type

        if droppath_mode == 'linear':
            droppaths = torch.linspace(0, droppath, depth)
        elif droppath_mode == 'constant':
            droppaths = [droppath] * depth
        else:
            raise ValueError

        if self.rope:
            self.rotary_embed = LlamaRotaryEmbedding(dim_in // attn_num_heads,
                                                     max_position_embeddings=1)

        self.blocks = nn.ModuleList([
            SqueezeformerBlock(stride=1,
                               layout=layout,
                               dim_in=dim_in,
                               dim_out=dim_out,
                               conv_ksize=conv_ksize,
                               conv_expand=conv_expand,
                               conv_depthwise=conv_depthwise,
                               conv_act=conv_act,
                               conv_type=conv_type,
                               conv_norm=conv_norm,
                               attn_type=attn_type,
                               attn_num_heads=attn_num_heads,
                               attn_dropout=attn_dropout,
                               attn_norm=attn_norm,
                               attn_pos_embed_type=attn_pos_embed_type,
                               ffn_expand=ffn_expand,
                               ffn_dropout=ffn_dropout,
                               ffn_act=ffn_act,
                               droppath=droppaths[block_idx],
                               max_position=max_position,
                               prenorm=prenorm) for block_idx in range(depth)
        ])

    def forward(self, x, mask=None):
        """
        Args:
            x: NLC
            mask: NL
        """
        N, L, C = x.shape

        if mask is None:
            mask = torch.full((N, L), True, dtype=torch.bool).to(x.device)

        # Cache cos, sin to save memory
        if self.rope:
            cos, sin = self.rotary_embed(x, L)
        else:
            cos, sin = None, None

        if self.alibi:
            alibi_bias = get_alibi(x.size(1), self.attn_num_heads).to(
                dtype=x.dtype, device=x.device).repeat(x.size(0), 1, 1, 1)
        else:
            alibi_bias = None

        outputs = []
        for block in self.blocks:
            x, mask = block(x, mask, sin=sin, cos=cos, alibi_bias=alibi_bias)
            outputs.append(x)
        return outputs


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


POOLS = {
    'avg': partial(nn.AdaptiveAvgPool1d, output_size=1),
    'max': partial(nn.AdaptiveMaxPool1d, output_size=1),
    'masked_max': layers.GlobalMaskedMaxPooling1d,
    'masked_avg': layers.GlobalMaskedAvgPooling1d,
    'masked_gem': partial(layers.GlobalMaskedGEMPooling1d)
}


class SqueezeformerModel(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model
        self.embedding = SqueezeformerEmbedding(cfg.vocab_size, cfg.dim,
                                                cfg.embedding.dropout,
                                                cfg.embedding.proj)
        self.encoder = SqueezeformerEncoder(
            depth=cfg.encoder.depth,
            layout=cfg.encoder.layout,
            dim_in=cfg.dim,
            dim_out=cfg.dim,
            conv_ksize=cfg.encoder.conv_ksize,
            conv_expand=cfg.encoder.conv_expand,
            conv_depthwise=cfg.encoder.conv_depthwise,
            conv_act=cfg.encoder.conv_act,
            conv_type=cfg.encoder.conv_type,
            conv_norm=cfg.encoder.conv_norm,
            attn_type=cfg.encoder.attn_type,
            attn_num_heads=cfg.encoder.attn_num_heads,
            attn_dropout=cfg.encoder.attn_dropout,
            attn_norm=cfg.encoder.attn_norm,
            attn_pos_embed_type=cfg.encoder.attn_pos_embed_type,
            ffn_expand=cfg.encoder.ffn_expand,
            ffn_dropout=cfg.encoder.ffn_dropout,
            ffn_act=cfg.encoder.ffn_act,
            droppath=cfg.encoder.droppath,
            max_position=1,
            prenorm=cfg.encoder.prenorm,
            droppath_mode=cfg.encoder.droppath_mode)

        if cfg.head.rnn:
            self.rnn = nn.GRU(cfg.dim,
                              cfg.dim,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=cfg.head.bi_rnn)
            head_dim = (2 if cfg.head.bi_rnn else 1) * cfg.dim
        else:
            self.rnn = None
            head_dim = cfg.dim

        self.is_masked_pool = ('masked' in cfg.pool_type)
        self.pool = POOLS[cfg.pool_type]()

        self.head = layers.MLP(head_dim,
                               cfg.head.mlp_chans + [3],
                               norm_layer=cfg.head.norm,
                               act_layer=get_act_fn(cfg.head.act),
                               dropout=cfg.head.dropout,
                               last_norm=False,
                               last_activation=False,
                               last_dropout=False)

    def forward(self, input_ids, mask):
        x = self.embedding(input_ids)
        x = self.encoder(x, mask)[-1]
        if self.rnn is not None:
            x, _h_n = self.rnn(x)
        x = x.permute(0, 2, 1)  # NLC -> NCL
        if self.is_masked_pool:
            x = self.pool(x, mask)
        else:
            x = self.pool(x).squeeze(-1)
        logits = self.head(x)
        return logits
