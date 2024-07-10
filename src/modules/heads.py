from timm.layers.classifier import ClassifierHead
from torch import nn
from src.modules import layers
from functools import partial
from typing import Callable, Optional, List
import torch


def LinearHead2d(in_dim, out_dim, pool_type, drop_rate):
    return ClassifierHead(
        in_features=in_dim,
        num_classes=out_dim,
        pool_type=pool_type,
        drop_rate=drop_rate,
        use_conv=False,
        input_fmt="NCHW",
    )


class LinearHead1d(nn.Module):

    def __init__(self, in_dim, out_dim, pool_type, drop_rate):
        super().__init__()
        assert pool_type in ['avg', 'max', 'masked_max', 'masked_avg']
        self.is_masked = ('masked' in pool_type)
        POOLS = {
            'avg': partial(nn.AdaptiveAvgPool1d, output_size=1),
            'max': partial(nn.AdaptiveMaxPool1d, output_size=1),
            'masked_max': layers.GlobalMaskedMaxPooling1d,
            'masked_mean': layers.GlobalMaskedAvgPooling1d,
        }
        pool_cls = POOLS[pool_type]
        self.pool = pool_cls()
        self.drop = nn.Dropout(p=drop_rate)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x, padding_mask=None):
        if self.is_masked:
            x = self.pool(x, padding_mask)
        else:
            x = self.pool(x)
        x = self.flatten(x)
        x = self.drop(x)
        x = self.fc(x)
        return x

