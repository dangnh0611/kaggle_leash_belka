# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
# https://pytorch.org/vision/main/_modules/torchvision/ops/stochastic_depth.html#stochastic_depth
# https://huggingface.co/spaces/Roll20/pet_score/blame/main/lib/timm/models/layers/drop.py
# @TODO: linear drop path by depth

import logging
from functools import partial
from typing import List, Tuple

import torch
from torch import nn

from src.modules.layers import (
    DepthwiseConv,
    ECAAttention,
    PointwiseConv,
    SEAttention,
    Downsample1d,
    MBBlock,
    MBBlockV2,
    LayerNorm1d,
)
from src.modules.misc import (
    get_act_fn,
    get_norm_layer,
    make_divisible,
    make_net_dims,
)
from src.modules import heads
from src.modules.stems import Stem

logger = logging.getLogger(__name__)


def make_cnn1d_specs(
    depth,
    dims,
    ksizes,
    num_blocks,
    activations,
    attentions,
    attn_start_idx = 0,
    block_name="MBBlock",
    first_stride = 2,
    downsample_method="strided_conv",
    flatten=False,
):
    assert downsample_method in ["strided_conv", "max_pool", "avg_pool"]

    def to_list(confs, depth):
        if isinstance(confs, (list, tuple)):
            assert len(confs) >= depth
            confs = confs[:depth]
        else:
            confs = [confs] * depth
        return confs

    ksizes = to_list(ksizes, depth)
    num_blocks = to_list(num_blocks, depth)
    activations = to_list(activations, depth)

    attns_is_str = isinstance(attentions, str)
    attentions = to_list(attentions, depth)
    if attns_is_str:
        # attns provided in form of a single str
        attentions[:attn_start_idx] = [None] * attn_start_idx


    specs = []
    for dim, ksize, nb, act, attn in zip(
        dims, ksizes, num_blocks, activations, attentions
    ):
        stage = []
        # downsample
        stage_idx = len(specs)
        if stage_idx == 0 and first_stride == 1:
            downsample_block = [block_name, dim, ksize, 1, act, attn]
        else:
            if downsample_method == "strided_conv":
                downsample_block = [block_name, dim, ksize, 2, act, attn]
            elif downsample_method in ["max_pool", "avg_pool"]:
                downsample_block = [downsample_method, 2, 2]
            else:
                raise ValueError
        stage.append(downsample_block)
        nb = nb - 1 if downsample_method == "strided_conv" else nb
        stage.extend([[block_name, dim, ksize, 1, act, attn]] * nb)
        specs.append(stage)
    if flatten:
        flatten_specs = []
        for stage in specs:
            flatten_specs.extend(stage)
        return flatten_specs
    else:
        return specs


def get_block(name):
    NAME2BLK = {
        "MBBlock": MBBlock,
        "MBBlockV2": MBBlockV2,
    }
    return NAME2BLK[name]


class Cnn1d(nn.Module):

    def __init__(
        self,
        in_dim,
        specs: List[Tuple[str, int, int, int, str, str]],
        expand_ratio=2.0,
        width_multiplier=1.0,
        dropout=0.0,
        droppath=0.0,
        round_nearest=8,
        norm_layer=None,
        downsample="avg",
    ):
        """Mobilenet v3 without stem on top."""
        super().__init__()

        if norm_layer is None:
            # momentum = 0.05: https://www.kaggle.com/code/hoyso48/1st-place-solution-training
            norm_layer = partial(nn.BatchNorm1d, eps=0.001, momentum=0.05)

        self.stages = nn.ModuleList()
        _in_dim = in_dim
        self._output_dims = [_in_dim]  # multi-scale dims
        for stage_idx, stage_specs in enumerate(specs):
            stage_blocks = []
            for _, blk_spec in enumerate(stage_specs):
                if blk_spec[0] == "max_pool":
                    _, pool_ksize, pool_stride = blk_spec
                    stage_blocks.append(nn.MaxPool1d(pool_ksize, pool_stride))
                elif blk_spec[0] == "avg_pool":
                    _, pool_ksize, pool_stride = blk_spec
                    stage_blocks.append(nn.AvgPool1d(pool_ksize, pool_stride))
                else:
                    blk_name, _out_dim, k, stride, act_name, attn_name = blk_spec
                    blk = get_block(blk_name)
                    _out_dim = make_divisible(
                        _out_dim * width_multiplier, round_nearest
                    )
                    stage_blocks.append(
                        blk(
                            _in_dim,
                            _out_dim,
                            k,
                            stride,
                            expand_ratio=expand_ratio,
                            downsample=downsample,
                            dropout=dropout,
                            droppath=droppath,
                            norm_layer=norm_layer,
                            act_layer=get_act_fn(act_name),
                            attn_name=attn_name,
                        )
                    )
                    _in_dim = _out_dim
            stage = nn.Sequential(*stage_blocks)
            self.stages.append(stage)
            self._output_dims.append(_out_dim)

        self._output_dim = _out_dim

        self.apply(self.init_weights)

    @property
    def depth(self):
        """Original input + number of stages"""
        return len(self.stages)

    @property
    def output_stride(self):
        return 2**self.depth

    @property
    def output_strides(self):
        return [2**i for i in range(self.depth + 1)]

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def output_dims(self):
        return self._output_dims

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

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_features(self, x):
        features = [x]
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


if __name__ == "__main__":
    N, C, T = 32, 20, 10000
    # blk_name, _out_dim, k, stride, act_name, attn_name
    # SPECS = [
    #     [("MBBlockV2", 64, 3, 2, "RELU", None)],
    #     [
    #         ("MBBlockV2", 64, 3, 2, "RELU", None),
    #         ("MBBlockV2", 64, 3, 1, "RELU", None),
    #         ("MBBlockV2", 64, 3, 1, "RELU", None),
    #     ],
    #     [
    #         ("MBBlockV2", 128, 3, 2, "RELU", None),
    #         ("MBBlockV2", 128, 3, 1, "RELU", "SE"),
    #         ("MBBlockV2", 128, 3, 1, "RELU", "SE"),
    #     ],
    #     [
    #         ("MBBlockV2", 256, 3, 2, "RELU", "SE"),
    #         ("MBBlockV2", 256, 3, 1, "HS", "ECA"),
    #         ("MBBlockV2", 256, 3, 1, "HS", "ECA"),
    #     ],
    #     [
    #         ("MBBlockV2", 512, 3, 2, "HS", "ECA"),
    #         ("MBBlockV2", 512, 3, 1, "GELU", None),
    #         ("MBBlockV2", 512, 3, 1, "GELU", None),
    #     ],
    # ]

    SPECS = [
        [
            ["MBBlock", 64, 3, 2, "RELU6", "ECA"],
            ["MBBlock", 64, 3, 1, "RELU6", "ECA"],
            ["MBBlock", 64, 3, 1, "RELU6", "ECA"],
        ],
        [
            ["MBBlock", 64, 3, 2, "RELU6", "ECA"],
            ["MBBlock", 64, 3, 1, "RELU6", "ECA"],
            ["MBBlock", 64, 3, 1, "RELU6", "ECA"],
        ],
        [
            ["MBBlock", 64, 3, 2, "RELU6", "ECA"],
            ["MBBlock", 64, 3, 1, "RELU6", "ECA"],
            ["MBBlock", 64, 3, 1, "RELU6", "ECA"],
        ],
        [
            ["MBBlock", 64, 3, 2, "RELU6", "ECA"],
            ["MBBlock", 64, 3, 1, "RELU6", "ECA"],
            ["MBBlock", 64, 3, 1, "RELU6", "ECA"],
        ],
        [
            ["MBBlock", 64, 3, 2, "RELU6", "ECA"],
            ["MBBlock", 64, 3, 1, "RELU6", "ECA"],
            ["MBBlock", 64, 3, 1, "RELU6", "ECA"],
        ],
        [
            ["MBBlock", 64, 3, 2, "RELU6", "ECA"],
            ["MBBlock", 64, 3, 1, "RELU6", "ECA"],
            ["MBBlock", 64, 3, 1, "RELU6", "ECA"],
        ],
    ]

    # SPECS = make_cnn1d_specs(depth = 6, dims=)

    NUM_PAD = 11

    import pprint

    print(pprint.pformat(SPECS, indent=4))

    dummy_input = torch.rand((N, C, T))
    dummy_padding_mask = (
        torch.Tensor([False] * (T - NUM_PAD) + [True] * NUM_PAD)
        .bool()
        .unsqueeze(0)
        .expand(N, T)
    )

    model = Cnn1d(
        in_dim=C,
        specs=SPECS,
        expand_ratio=2.0,
        width_multiplier=1.0,
        dropout=0.0,
        droppath=0.0,
        round_nearest=8,
        norm_layer=None,
        downsample="avg",
    )
    print(model)
    with torch.no_grad():
        features = model.forward(dummy_input)
        print("\n\n---------------------------------------\n")
        print("Input shape:", dummy_input.shape)
        for i, feat in enumerate(features):
            print(f"Output shape #{i}:", feat.shape)
