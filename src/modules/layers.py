from torch import nn
import torch
from torch.nn import functional as F

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init
import math
from functools import partial
import math
from typing import List, Tuple
from src.modules.misc import make_divisible
from torchvision.ops import StochasticDepth
from typing import Callable, Optional, List
import torch
import torch as th
import numpy as np


class TransposeLast(nn.Module):

    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)


class MaskedSoftmax(nn.Module):

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def forward(self, inputs, mask=None):
        if mask is not None:
            inputs = inputs.masked_fill(~mask, torch.finfo(inputs.dtype).min)
        return F.softmax(inputs, dim=self.dim)  #, dtype=torch.float32)


# Modify from torchvision's MLP, add last Activation before dropout
class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        act_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
        last_norm=True,
        last_activation=True,
        last_dropout=True,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for i, hidden_dim in enumerate(hidden_channels):
            is_not_last = (i != len(hidden_channels) - 1)
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None and (is_not_last or last_norm):
                layers.append(norm_layer(hidden_dim))
            if is_not_last or last_activation:
                layers.append(act_layer())
            if is_not_last or last_dropout:
                layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        super().__init__(*layers)


class GEMPooling2d(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p),
                            (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'


class GEMPooling1d(nn.Module):

    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(x.clamp(min=eps).pow(p),
                            self.kernel_size).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'


class GlobalMaskedGEMPooling1d(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self._avg_pool = GlobalMaskedAvgPooling1d()

    def forward(self, x, padding_mask=None):
        x = x.clamp(min=self.eps).pow(self.p)
        x = self._avg_pool(x, padding_mask)
        x = x.pow(1. / self.p)
        return x


# https://github.com/amedprof/Feedback-Prize--English-Language-Learning/blob/main/src/model_zoo/pooling.py
class GlobalMaskedAvgPooling1d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_state, padding_mask=None):
        """
        hidden_state: NCT
        padding_mask: NT (bool, True is non-pad, False is pad)
        """
        if padding_mask is None:
            return F.adaptive_avg_pool1d(hidden_state, 1)
        padding_mask_expanded = padding_mask.unsqueeze(1).expand(
            hidden_state.size())  # NCT
        sum_embeddings = torch.sum(hidden_state * padding_mask_expanded,
                                   -1)  # NCT -> NC
        # @TODO (dangnh): optimize
        sum_mask = padding_mask_expanded.sum(-1)  # NCT -> NC
        # sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask  # NC
        return mean_embeddings


class GlobalMaskedMaxPooling1d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_state, padding_mask=None):
        """
        hidden_state: NCT
        padding_mask: NT (bool, True (1) is non-pad, False (0) is pad)
        """
        if padding_mask is None:
            return F.adaptive_max_pool1d(hidden_state, 1)
        padding_mask_expanded = torch.logical_not(padding_mask).unsqueeze(
            1).expand(hidden_state.size())  # NCT
        hidden_state_copy = hidden_state.clone()
        hidden_state_copy.masked_fill(padding_mask_expanded, float('-inf'))
        return torch.max(hidden_state_copy, -1)[0]  # NCT -> NC


class Conv1dNormAct(nn.Sequential):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            act_layer=partial(nn.ReLU, inplace=True),
            norm_layer=partial(nn.BatchNorm1d),
    ):
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=(norm_layer is None),
        )
        act = act_layer()
        if norm_layer is None:
            norm = nn.Identity()
        else:
            norm = norm_layer(out_channels)
        super().__init__(conv, norm, act)


class SameConv1d(nn.Conv1d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        if isinstance(kernel_size, int) and isinstance(dilation, int):
            padding = (kernel_size - 1) // 2 * dilation
        else:
            raise ValueError
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )


class CausalConv1d(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            nn.ConstantPad1d((dilation * (kernel_size - 1), 0), 0.),
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      padding='valid',
                      dilation=dilation,
                      groups=groups,
                      bias=bias))


class DWConv1d(nn.Conv1d):

    def __init__(
        self,
        in_channels,
        depth_multiplier=1,
        kernel_size=3,
        stride=1,
        dilation=1,
        bias=True,
    ):
        assert stride <= 2
        if isinstance(kernel_size, int) and isinstance(dilation, int):
            padding = (kernel_size - 1) // 2 * dilation
        else:
            raise ValueError
        super().__init__(
            in_channels,
            depth_multiplier * in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )


class PointwiseConv(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 norm_layer=None,
                 act_layer=None,
                 bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_dim,
                              out_dim,
                              kernel_size=1,
                              stride=1,
                              bias=bias)
        self.norm = norm_layer(
            out_dim) if norm_layer is not None else nn.Identity()
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DepthwiseConv(nn.Module):

    def __init__(self,
                 in_dim,
                 kernel_size,
                 stride,
                 norm_layer,
                 act_layer=None,
                 bias=True):
        super().__init__()
        self.conv = DWConv1d(
            in_dim,
            depth_multiplier=1,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )
        self.norm = norm_layer(in_dim)
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/SEAttention.py
class SEAttention(nn.Module):

    def __init__(self,
                 inp_dim,
                 hidden_dim,
                 hidden_act=nn.ReLU,
                 scale_act=nn.Sigmoid):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim, bias=False),
            hidden_act(inplace=True),
            nn.Linear(hidden_dim, inp_dim, bias=False),
            scale_act(),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: NCT
        """
        y = self.gap(x).squeeze(-1)  # NC
        y = self.fc(y).unsqueeze(-1)  # NC -> NC -> NC1
        return x * y.expand_as(x)  # NCT


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3, bias=True):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=bias)
        self.act = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: NCT
        """
        # NCT
        y = self.gap(x).permute(0, 2, 1)  # NCT -> NC1 -> N1C
        y = self.act(self.conv(y)).permute(0, 2, 1)  # N1C -> N1C -> NC1
        return x * y.expand_as(x)  # NCT


class LayerNorm1d(nn.Module):

    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        return x


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding module.
    Args:
        d_model: Embedding dimension.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : Input tensor B X T X C
        Returns:
            torch.Tensor: Encoded tensor B X T X C
        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1:self.pe.size(1) // 2 +
            x.size(1),
        ]
        return pos_emb


# https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/padding.py
# Can SAME padding for given args be done statically?
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int,
                stride: int = 1,
                dilation: int = 1,
                **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == "same":
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == "valid":
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


# tweak from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pool2d_same.py#L56
def create_pool1d(pool_type, kernel_size, stride=None, **kwargs):
    stride = stride or kernel_size
    padding = kwargs.pop("padding", "")
    padding, is_dynamic = get_padding_value(padding,
                                            kernel_size,
                                            stride=stride,
                                            **kwargs)
    if is_dynamic:
        # never goes here
        raise NotImplementedError
        if pool_type == "avg":
            return AvgPool2dSame(kernel_size, stride=stride, **kwargs)
        elif pool_type == "max":
            return MaxPool2dSame(kernel_size, stride=stride, **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"
    else:
        if pool_type == "avg":
            return nn.AvgPool1d(kernel_size,
                                stride=stride,
                                padding=padding,
                                **kwargs)
        elif pool_type == "max":
            return nn.MaxPool1d(kernel_size,
                                stride=stride,
                                padding=padding,
                                **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"


# tweak from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/maxxvit.py#L303
class Downsample1d(nn.Module):
    """A downsample pooling module supporting several maxpool and avgpool modes
    * 'max' - MaxPool2d w/ kernel_size 3, stride 2, padding 1
    * 'max2' - MaxPool2d w/ kernel_size = stride = 2
    * 'avg' - AvgPool2d w/ kernel_size 3, stride 2, padding 1
    * 'avg2' - AvgPool2d w/ kernel_size = stride = 2
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        pool_type: str = "avg2",
        padding: str = "",
        bias: bool = True,
    ):
        super().__init__()
        assert pool_type in ("max", "max2", "avg", "avg2", "conv", "conv2")
        if pool_type == "max":
            self.pool = create_pool1d("max",
                                      kernel_size=3,
                                      stride=2,
                                      padding=padding or 1)
        elif pool_type == "max2":
            self.pool = create_pool1d("max", 2, padding=padding
                                      or 0)  # kernel_size == stride == 2
        elif pool_type == "avg":
            self.pool = create_pool1d(
                "avg",
                kernel_size=3,
                stride=2,
                count_include_pad=False,
                padding=padding or 1,
            )
        elif pool_type == "avg2":
            self.pool = create_pool1d("avg", 2, padding=padding or 0)
        elif pool_type == "conv":
            self.pool = nn.Conv1d(dim,
                                  dim_out,
                                  kernel_size=3,
                                  stride=2,
                                  padding=0)
        elif pool_type == "conv2":
            self.pool = nn.Conv1d(dim,
                                  dim_out,
                                  kernel_size=2,
                                  stride=2,
                                  padding=1)
        else:
            raise NotImplementedError

        if dim != dim_out and "conv" not in pool_type:
            self.expand = nn.Conv1d(dim, dim_out, 1, bias=bias)
        else:
            self.expand = nn.Identity()

    def forward(self, x):
        x = self.pool(x)  # spatial downsample
        x = self.expand(x)  # expand chs
        return x


class MBBlock(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride,
        expand_ratio,
        downsample="avg",
        dropout=0.0,
        droppath=0.0,
        norm_layer=None,
        act_layer=None,
        attn_name=None,
    ):
        super(MBBlock, self).__init__()
        self.stride = stride

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if act_layer is None:
            act_layer = nn.ReLU

        hidden_dim = int(round(in_dim * expand_ratio))

        self.use_shortcut = True
        if self.stride == 1:
            assert in_dim == out_dim
            self.shortcut = nn.Identity()
        elif self.stride == 2 and downsample != "none":
            self.shortcut = Downsample1d(in_dim,
                                         out_dim,
                                         downsample,
                                         bias=True)
        elif self.stride == 2 and downsample == "none":
            self.use_shortcut = False
        else:
            raise ValueError

        if expand_ratio != 1:
            # expand pw
            self.expand_pw = PointwiseConv(in_dim,
                                           hidden_dim,
                                           norm_layer=norm_layer,
                                           act_layer=act_layer)
        else:
            raise NotImplementedError
        # depth-wise
        self.dw = DepthwiseConv(
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        if attn_name == "SE":
            se_hidden_dim = make_divisible(hidden_dim // 4, 8)
            # mobilenetv3 use HardSigmoid instead of Sigmoid
            # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py#L52C35-L52C35
            self.attn = SEAttention(hidden_dim,
                                    se_hidden_dim,
                                    hidden_act=nn.ReLU,
                                    scale_act=nn.Hardsigmoid)
        elif attn_name == "ECA":
            self.attn = ECAAttention(kernel_size=5)
        elif attn_name is None:
            self.attn = None
        else:
            raise ValueError(f"Invalid channel attention {attn_name}")
        # point-wise + norm, no activation (linear)
        self.pw = PointwiseConv(hidden_dim,
                                out_dim,
                                norm_layer=norm_layer,
                                act_layer=None)
        self.droppath = StochasticDepth(p=droppath, mode="row")
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.use_shortcut:
            skip = self.shortcut(x)
        x = self.expand_pw(x)
        x = self.dw(x)
        if self.attn is not None:
            x = self.attn(x)
        x = self.pw(x)
        x = self.droppath(self.dropout(x))
        if self.use_shortcut:
            x = x + skip
        return x

    def init_weights(self, m):
        # weight initialization
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# less activations and norms
# https://www.kaggle.com/code/hoyso48/1st-place-solution-inference
class MBBlockV2(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride,
        expand_ratio,
        downsample="avg",
        dropout=0.0,
        droppath=0.0,
        norm_layer=None,
        act_layer=None,
        attn_name=None,
    ):
        super(MBBlockV2, self).__init__()
        self.stride = stride

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if act_layer is None:
            act_layer = nn.ReLU

        hidden_dim = int(round(in_dim * expand_ratio))

        if self.stride == 1:
            assert in_dim == out_dim
            self.shortcut = nn.Identity()
        elif self.stride == 2:
            self.shortcut = Downsample1d(in_dim,
                                         out_dim,
                                         downsample,
                                         bias=True)
        else:
            raise ValueError

        if expand_ratio != 1:
            # expand pw
            # with bias + activation, no norm
            self.expand_pw = PointwiseConv(in_dim,
                                           hidden_dim,
                                           norm_layer=None,
                                           act_layer=act_layer,
                                           bias=True)
        else:
            raise NotImplementedError
        # depth-wise
        # ori: no bias + batchnorm, no activation
        # this: bias + BN/LN/GN, no activation
        # bias could be rebundant but no impact on performance -> keep bias=True
        self.dw = DepthwiseConv(
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            norm_layer=norm_layer,
            act_layer=None,
            bias=True,
        )
        if attn_name == "SE":
            se_hidden_dim = make_divisible(hidden_dim // 4, 8)
            # mobilenetv3 use HardSigmoid instead of Sigmoid
            # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py#L52C35-L52C35
            self.attn = SEAttention(hidden_dim,
                                    se_hidden_dim,
                                    hidden_act=nn.ReLU,
                                    scale_act=nn.Hardsigmoid)
        elif attn_name == "ECA":
            self.attn = ECAAttention(kernel_size=5)
        elif attn_name is None:
            self.attn = None
        else:
            raise ValueError(f"Invalid channel attention {attn_name}")
        # point-wise, no norm, no activation (linear)
        self.pw = PointwiseConv(hidden_dim,
                                out_dim,
                                norm_layer=None,
                                act_layer=None,
                                bias=True)
        self.droppath = StochasticDepth(p=droppath, mode="row")
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        skip = self.shortcut(x)
        x = self.expand_pw(x)
        x = self.dw(x)
        if self.attn is not None:
            x = self.attn(x)
        x = self.pw(x)
        x = self.droppath(self.dropout(x))
        x = x + skip
        return x

    def init_weights(self, m):
        # weight initialization
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class AttentionPooling1d(nn.Module):
    """
    Ref: https://github.com/daniel-code/TubeViT/blob/main/tubevit/model.py
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf

    code from https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
    """

    def __init__(self, input_dim, channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        self.proj = nn.Linear(input_dim, 1)
        self.masked_softmax = MaskedSoftmax(-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: NCL if channel_first, else NLC (default)
            mask: NL, True -> nonpad, False -> pad
        Returns:
            NC
        """
        if self.channel_first:
            # NCL -> NLC
            x = x.permute(0, 2, 1)
        _tmp = self.proj(x).squeeze(dim=-1)
        att_w = self.masked_softmax(_tmp, mask).unsqueeze(dim=-1)
        x = torch.sum(x * att_w, dim=1)
        return x


class ConcatAttentionPooling1d(nn.Module):
    """
    Concat of Attention Pooling + CLS Pooling
    """

    def __init__(self, input_dim, channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        self.proj = nn.Linear(input_dim, 1)
        self.masked_softmax = MaskedSoftmax(-1)

    def forward(self, input, mask=None):
        """
        Args:
            x: NCL if channel_first, else NLC (default)
            mask: NL, True -> nonpad, False -> pad
        Returns:
            NC
        """
        x = input[:, 1:, :]
        mask = mask[:, 1:]
        if self.channel_first:
            # NCL -> NLC
            x = x.permute(0, 2, 1)
        _tmp = self.proj(x).squeeze(dim=-1)
        att_w = self.masked_softmax(_tmp, mask).unsqueeze(dim=-1)
        x = torch.sum(x * att_w, dim=1)
        x = torch.cat([input[:, 0, :], x], axis=-1)  # N x (2C)
        return x
