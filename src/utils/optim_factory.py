import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)


def resolve_optimizer_kwargs(cfg):
    kwargs = dict(cfg)
    return kwargs


def create_optimizer(model: nn.Module, cfg):
    optim_kwargs = resolve_optimizer_kwargs(cfg)
    name = optim_kwargs.pop("name")
    logger.info('Optim kwargs: %s', optim_kwargs)

    if name.startswith("timm@"):
        optim_kwargs["opt"] = name.replace("timm@", "")
        from timm.optim.optim_factory import create_optimizer_v2
        optimizer = create_optimizer_v2(model, **optim_kwargs)
    elif name.startswith("torch@"):
        name = name.replace("torch@", "")
        TORCH_OPTIMS = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
        }
        optim_cls = TORCH_OPTIMS[name]
        optimizer = optim_cls(model.parameters(), **optim_kwargs)
    elif name.startswith("apex@"):
        name = name.replace("apex@", "")
        import apex
        APEX_OPTIMS = {
            "lamb": apex.optimizers.FusedLAMB,
            "mixed_lamb": apex.optimizers.FusedMixedPrecisionLamb
        }
        optim_cls = APEX_OPTIMS[name]
        optimizer = optim_cls(model.parameters(), **optim_kwargs)
    else:
        raise ValueError

    return optimizer
