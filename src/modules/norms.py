from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from functools import partial
import math
# Masked Batch Normalization


def masked_batch_norm(input: Tensor,
                      mask: Tensor,
                      weight: Optional[Tensor],
                      bias: Optional[Tensor],
                      running_mean: Optional[Tensor],
                      running_var: Optional[Tensor],
                      training: bool,
                      momentum: float,
                      eps: float = 1e-5) -> Tensor:
    #from https://gist.github.com/yangkky/364413426ec798589463a3a88be24219
    r"""Applies Masked Batch Normalization for each channel in each data sample in a batch.
    See :class:`~MaskedBatchNorm1d`, :class:`~MaskedBatchNorm2d`, :class:`~MaskedBatchNorm3d` for details.
    """
    if not training and (running_mean is None or running_var is None):
        raise ValueError(
            'Expected running_mean and running_var to be not None when training=False'
        )

    num_dims = len(input.shape[2:])
    _dims = (0, ) + tuple(range(-num_dims, 0))
    _slice = (None, ...) + (None, ) * num_dims

    if training:
        num_elements = mask.sum(_dims)
        mean = (input * mask).sum(_dims) / num_elements  # (C,)
        var = (((input - mean[_slice]) * mask)**
               2).sum(_dims) / num_elements  # (C,)

        if running_mean is not None:
            running_mean.copy_(running_mean * (1 - momentum) +
                               momentum * mean.detach())
        if running_var is not None:
            running_var.copy_(running_var * (1 - momentum) +
                              momentum * var.detach())
    else:
        mean, var = running_mean, running_var

    out = (input - mean[_slice]) / torch.sqrt(var[_slice] + eps)  # (N, C, ...)

    if weight is not None and bias is not None:
        out = out * weight[_slice] + bias[_slice]

    return out


class _MaskedBatchNorm(_BatchNorm):
    #from https://gist.github.com/yangkky/364413426ec798589463a3a88be24219
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(_MaskedBatchNorm, self).__init__(num_features, eps, momentum,
                                               affine, track_running_stats)

    def forward(self, input: Tensor, mask: Tensor = None) -> Tensor:
        self._check_input_dim(input)
        if mask is not None:
            self._check_input_dim(mask)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var
                                                           is None)
        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if mask is None:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean
                if not self.training or self.track_running_stats else None,
                self.running_var
                if not self.training or self.track_running_stats else None,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps)
        else:
            return masked_batch_norm(
                input, mask, self.weight, self.bias, self.running_mean
                if not self.training or self.track_running_stats else None,
                self.running_var
                if not self.training or self.track_running_stats else None,
                bn_training, exponential_average_factor, self.eps)


class MaskedBatchNorm1d(torch.nn.BatchNorm1d, _MaskedBatchNorm):
    #from https://gist.github.com/yangkky/364413426ec798589463a3a88be24219
    r"""Applies Batch Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..
    See documentation of :class:`~torch.nn.BatchNorm1d` for details.
    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 channels_last: bool = False) -> None:
        super(MaskedBatchNorm1d, self).__init__(num_features, eps, momentum,
                                                affine, track_running_stats)
        self.channels_last = channels_last

    def forward(self, inputs, mask=None):
        if self.channels_last:
            inputs = inputs.permute(0, 2, 1)
        if mask is not None:
            mask = mask[:, None, :]
        out = super(MaskedBatchNorm1d, self).forward(inputs, mask)
        if self.channels_last:
            out = out.permute(0, 2, 1)
        return out


class MaskedBatchNorm1dV2(nn.BatchNorm1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, mask=None):
        """
        Args:
            x: NCL
            mask: NL where 1 is non-mask, 0 is mask
        """
        if mask is None:
            return super().forward(x)

        # NCL -> NLC -> (NL)C
        N, C, L = x.shape
        x = x.permute(0, 2, 1).reshape(-1, C)
        mask = mask.view(-1)
        x[mask] = super().forward(x[mask])
        x = x.view((N, L, C)).permute(0, 2, 1)  # NLC -> NCL
        return x


class LayerScale(nn.Module):
    """
    Computes an affine transformation y = x * scale + bias, either learned via adaptive weights, or fixed.
    Efficient alternative to LayerNorm where we can avoid computing the mean and variance of the input, and
    just rescale the output of the previous layer.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.tensor([1.] * dim)[None, None, :])
        self.bias = torch.nn.Parameter(torch.tensor([0.] * dim)[None, None, :])

    def forward(self, x):
        return x * self.scale + self.bias


class IBNorm2d(nn.Module):
    """
    Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Ref: https://github.com/XingangPan/IBN-Net/blob/master/ibnnet/modules.py

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes, ratio=0.5):
        super().__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class IBNorm1d(nn.Module):

    def __init__(self, planes, ratio=0.5):
        super().__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm1d(self.half, affine=True)
        self.BN = nn.BatchNorm1d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class GhostBatchNorm1d(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self,
                 input_dim,
                 vbs=None,
                 splits=None,
                 eps=1e-05,
                 momentum=0.01,
                 affine=True,
                 track_running_stats=True,
                 device=None,
                 dtype=None):
        """
        Args:
            input_dim: input channel dim size
            vbs: virtual batch size
        """
        super().__init__()
        self.input_dim = input_dim
        if vbs is not None:
            assert splits is None
        elif splits is not None:
            assert vbs is None
        else:
            raise ValueError
        self.vbs = vbs
        self.bn = nn.BatchNorm1d(self.input_dim,
                                 eps=eps,
                                 momentum=momentum,
                                 affine=affine,
                                 track_running_stats=track_running_stats,
                                 device=device,
                                 dtype=dtype)

    def forward(self, x):
        chunks = x.chunk(int(math.ceil(x.size(0) / self.vbs)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)


class ChannelLastBatchNorm1d(nn.BatchNorm1d):
    """
    Input: NLC
    """

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


class ChannelLastMaskedBatchNorm1d(MaskedBatchNorm1d):
    """
    Quick implementation and lack of optimization.
    Input: NLC
    """

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


class ChannelFirstLayerNorm1d(nn.LayerNorm):
    """
    Input: NCL
    """

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


class ChannelFirstLayerNorm2d(nn.LayerNorm):
    """
    Input: NCHW
    """

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return x


class ChannelLastInstanceNorm1d(nn.InstanceNorm1d):
    """
    Input: NCHW
    """

    def forward(self, x):
        x = x.permute(0, 2, 1)  # NCHW -> NHWC
        x = super().forward(x)
        x = x.permute(0, 2, 1)  # NHWC -> NCHW
        return x


# @TODO: support GroupNorm
# @TODO: support SplitBatchNorm
# @TODO: support GhostBatchNorm
# @TODO: flexible args & kwargs
def get_norm_layer(name, *args, **kwargs):
    if name == 'batchnorm_1d_first':
        # Channel-first BatchNorm1d, NCL
        return partial(nn.BatchNorm1d, *args, **kwargs)
    elif name == 'batchnorm_1d_last':
        # Channel-last BatchNorm1d, NLC
        return partial(ChannelLastBatchNorm1d, *args, **kwargs)
    elif name == 'masked_batchnorm_1d_first':
        # Channel-first Masked BatchNorm1d, NCL
        # mask=0 is pad, mask=1 is non-pad
        return partial(MaskedBatchNorm1d, *args, **kwargs)
    elif name == 'masked_batchnorm_1d_last':
        # Channel-last Masked BatchNorm1d, NLC
        # mask=0 is pad, mask=1 is non-pad
        return partial(ChannelLastMaskedBatchNorm1d, *args, **kwargs)
    elif name == 'masked_batchnorm_1d_first_v2':
        # Channel-first Masked BatchNorm1d, NCL
        # mask=0 is pad, mask=1 is non-pad
        return partial(MaskedBatchNorm1dV2, *args, **kwargs)
    elif name == 'layernorm_1d_first':
        # Channel-first LayerNorm1d, NCL
        return partial(ChannelFirstLayerNorm1d, *args, **kwargs)
    elif name == 'layernorm_1d_last':
        # Channel-last LayerNorm1d, NLC
        return partial(nn.LayerNorm, *args, **kwargs)
    elif name == 'instancenorm_1d_first':
        # Channel-first InstanceNorm1d, NCL
        return partial(nn.InstanceNorm1d, *args, **kwargs)
    elif name == 'instancenorm_1d_last':
        # Channel-last InstanceNorm1d, NLC
        return partial(ChannelLastInstanceNorm1d, *args, **kwargs)
    elif name == 'ibnorm_1d_first':
        # Channel-first IBN, NCL
        return partial(IBNorm1d, *args, **kwargs)
    elif name == 'ibnorm_1d_last':
        # Channel-last IBN, NLC
        raise NotImplementedError
    else:
        raise ValueError
