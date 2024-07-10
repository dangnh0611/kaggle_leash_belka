from collections import OrderedDict

import torch
from torch import nn

from src.modules.layers import Conv1dNormAct, DWConv1d, SameConv1d
from src.modules.misc import make_divisible


class PatchEmbedStem(nn.Sequential):

    def __init__(self, in_dim, out_dim, norm_layer, patch_size=4, bias=True):
        return super().__init__(
            SameConv1d(
                in_dim, out_dim, kernel_size=patch_size, stride=patch_size, bias=bias
            ),
            norm_layer(out_dim),
        )


class SqueezeStem(nn.Module):
    """Squeezeformer like"""

    def __init__(
        self, in_dim: int, out_dim: int, kernel_size=3, stride=2, depth=2
    ) -> None:
        super().__init__()
        # assert depth >= 2
        layers = [
            SameConv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
        ]
        layers.extend(
            [
                DWConv1d(out_dim, depth_multiplier=1, kernel_size=3, stride=1),
                nn.ReLU(),
            ]
            * (depth - 1)
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/maxxvit.py#L1068
class MaxxvitStem(nn.Module):
    """Maxxvit like"""

    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride,
        norm_layer,
        act_layer,
        preact=False,
        bias=True,
    ):
        super().__init__()
        self.conv1 = SameConv1d(in_dim, out_dim, kernel_size, stride=stride, bias=bias)
        self.norm1 = norm_layer(out_dim)
        self.act1 = act_layer()
        self.conv2 = SameConv1d(out_dim, out_dim, 3, stride=1, bias=bias)
        self.norm2 = norm_layer(out_dim) if not preact else nn.Identity()
        self.act2 = act_layer() if not preact else nn.Identity()

    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnetv2.py#L279
class ResnetV2Stem(nn.Module):
    """Resnetv2 (actually, Resnet) like"""

    def __init__(
        self,
        in_dim,
        out_dim,
        norm_layer,
        kernel_size=3,
        stride=2,
        stem_type="",
        preact=False,
    ) -> None:
        super().__init__()
        layers = OrderedDict()
        assert stem_type in ("res", "deep", "deep_tiered")

        # NOTE conv padding mode can be changed by overriding the conv_layer def
        if "deep" in stem_type:
            # A 3 deep 3x3  conv stack as in ResNet V1D models
            if "tiered" in stem_type:
                stem_chs = (
                    kernel_size * out_dim // 8,
                    out_dim // 2,
                )  # 'T' resnets in resnet.py
            else:
                stem_chs = (out_dim // 2, out_dim // 2)  # 'D' ResNets
            layers["conv1"] = SameConv1d(
                in_dim, stem_chs[0], kernel_size=kernel_size, stride=stride
            )
            layers["norm1"] = norm_layer(stem_chs[0])
            layers["conv2"] = SameConv1d(
                stem_chs[0], stem_chs[1], kernel_size=3, stride=1
            )
            layers["norm2"] = norm_layer(stem_chs[1])
            layers["conv3"] = SameConv1d(stem_chs[1], out_dim, kernel_size=3, stride=1)
            if not preact:
                layers["norm3"] = norm_layer(out_dim)
        elif stem_type == "res":
            # The usual 7x7 stem conv
            layers["conv"] = SameConv1d(in_dim, out_dim, kernel_size, stride=stride)
            if not preact:
                layers["norm"] = norm_layer(out_dim)
        self.layers = nn.Sequential(layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class ConvNeXtStem(nn.Module):
    """Resnetv2 (actually, Resnet) like"""

    def __init__(
        self,
        in_dim,
        out_dim,
        norm_layer,
        kernel_size=3,
        stride=2,
        stem_type="overlap",
        preact=False,
    ) -> None:
        super().__init__()
        layers = OrderedDict()
        assert stem_type in ("patch", "overlap", "overlap_tiered")

        if stem_type == "patch":
            # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
            self.layers = PatchEmbedStem(in_dim, out_dim, norm_layer, stride, bias=True)
        else:
            mid_dim = (
                make_divisible(out_dim // 2, 8) if "tiered" in stem_type else out_dim
            )
            self.layers = nn.Sequential(
                SameConv1d(
                    in_dim, mid_dim, kernel_size=kernel_size, stride=stride, bias=True
                ),
                SameConv1d(mid_dim, out_dim, kernel_size=3, stride=1, bias=True),
                norm_layer(out_dim),
            )

    def forward(self, x):
        x = self.layers(x)
        return x


def Stem(
    in_dim,
    out_dim,
    stem_type,
    norm_layer,
    act_layer,
    kernel_size=3,
    stride=2,
    depth=2,
    preact=False,
):
    assert stem_type in [
        "patch",
        "squeeze",
        "res",
        "deep",
        "deep_tiered",
        "overlap",
        "overlap_tiered",
        "maxxvit",
    ]

    if stem_type == "patch":
        return PatchEmbedStem(in_dim, out_dim, norm_layer, stride, bias=True)
    elif stem_type == "squeeze":
        return SqueezeStem(in_dim, out_dim, kernel_size, stride, depth)
    elif stem_type in ["res", "deep", "deep_tiered"]:
        return ResnetV2Stem(
            in_dim, out_dim, norm_layer, kernel_size, stride, stem_type, preact
        )
    elif stem_type in ["overlap", "overlap_tiered"]:
        return ConvNeXtStem(
            in_dim, out_dim, norm_layer, kernel_size, stride, stem_type, preact
        )
    elif stem_type == "maxxvit":
        return MaxxvitStem(
            in_dim, out_dim, kernel_size, stride, norm_layer, act_layer, preact
        )
    else:
        raise ValueError
