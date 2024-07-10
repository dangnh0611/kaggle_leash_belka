import torch
import torch.nn as nn
from src.modules.kan import KAN
from src.modules.misc import get_norm_layer

NORMS = {'LN': nn.LayerNorm}


class KANModel(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model
        self.kan = KAN(
            in_channels=cfg.in_dim,
            hidden_channels=cfg.hidden_dims + [3],
            norm_layer=NORMS[cfg.norm] if cfg.norm is not None else None,
            kan_type=cfg.kan_type,
            dropout=cfg.dropout,
            last_norm=False,
            last_dropout=False,
            last_linear=False,
            first_linear=False,
            grid=cfg.grid,
            k=cfg.k,
        )

    def forward(self, x, padding_mask=None):
        x = self.kan(x.float())
        return x
