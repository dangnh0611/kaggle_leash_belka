import torch
import torch.nn as nn
from src.modules.kan import KAN

NORMS = {'LN': nn.LayerNorm}


class BBKANModel(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model
        self.embed = nn.Embedding(2110, cfg.in_dim, padding_idx=None)
        self.kan = KAN(
            in_channels=3 * cfg.in_dim,
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
        x = self.embed(x)
        N, L, C = x.shape
        assert L == 3
        x = x.reshape(N, L * C)
        x = self.kan(x.float())
        return x
