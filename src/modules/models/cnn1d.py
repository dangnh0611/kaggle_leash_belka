# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
# https://pytorch.org/vision/main/_modules/torchvision/ops/stochastic_depth.html#stochastic_depth
# https://huggingface.co/spaces/Roll20/pet_score/blame/main/lib/timm/models/layers/drop.py
# @TODO: linear drop path by depth

import logging
from functools import partial
from typing import List, Tuple

import torch
from torch import nn

from src.modules.misc import (
    get_act_fn,
    get_norm_layer,
    make_divisible,
    make_net_dims,
)
from src.modules import heads
from src.modules.stems import Stem
from src.modules.backbones.cnn1d import make_cnn1d_specs, Cnn1d

logger = logging.getLogger(__name__)


class Model1d(nn.Module):
    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model

        # EMBEDDING
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=None)

        # STEM
        stem_dim = make_divisible(
            int(cfg.stem.out_dim * cfg.backbone.width_multiplier), 8
        )
        self.stem = Stem(
            cfg.embed_dim,
            stem_dim,
            cfg.stem.name,
            get_norm_layer(cfg.stem.norm, eps=0.001, momentum=0.05),
            get_act_fn(cfg.stem.act),
            cfg.stem.ksize,
            cfg.stem.stride,
            cfg.stem.depth,
            cfg.stem.preact,
        )

        # BACKBONE
        if cfg.backbone.name == "cnn1d":
            # build 1D CNN
            cnn_dims = make_net_dims(
                cfg.backbone.depth,
                cfg.stem.out_dim,
                cfg.backbone.dim_scale_method,
                width_multiplier=1,
                divisor=8,
            )
            cnn_specs = make_cnn1d_specs(
                cfg.backbone.depth,
                cnn_dims,
                cfg.backbone.ksize,
                cfg.backbone.blocks_per_stage,
                cfg.backbone.act,
                cfg.backbone.attns,
                cfg.backbone.attn_start_idx,
                cfg.backbone.block,
                downsample_method=cfg.backbone.downsample,
                first_stride = cfg.backbone.first_stride,
                flatten=False,
            )
            import pprint
            logger.info('CNN 1D SPECS:\n%s', pprint.pformat(cnn_specs, indent=4))
            self.backbone = Cnn1d(
                in_dim=cfg.stem.out_dim,
                specs=cnn_specs,
                expand_ratio=cfg.backbone.expand_ratio,
                width_multiplier=cfg.backbone.width_multiplier,
                dropout=cfg.backbone.drop_rate,
                droppath=cfg.backbone.drop_path_rate,
                round_nearest=8,
                norm_layer=get_norm_layer(cfg.backbone.norm, eps=0.001, momentum=0.05),
                downsample=cfg.backbone.shortcut_downsample,
            )
        else:
            raise ValueError

        # HEAD
        if cfg.head.name == "linear_1d":
            self.head = heads.LinearHead1d(
                in_dim=self.backbone.output_dim,
                out_dim=cfg.head.num_classes,
                pool_type=cfg.head.pool_type,
                drop_rate=cfg.head.drop_rate,
            )
        else:
            raise ValueError

    def forward(self, x, padding_mask = None):
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.backbone(x)
        x = self.head(x, padding_mask)
        return x



