""" Exponential Moving Average (EMA) of model updates

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple, Optional
from timm.utils.distributed import distribute_bn

logger = logging.getLogger(__name__)


class ModelEma:
    """ Model Exponential Moving Average (DEPRECATED)

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This version is deprecated, it does not work with scripted models. Will be removed eventually.

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device='', resume=''):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                # ema model may have been wrapped by DataParallel, and need module prefix
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)
            logger.info("Loaded state_dict_ema")
        else:
            logger.warning("Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model):
        # correct a mismatch in state dict keys
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)


class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model: nn.Module, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module: nn.Module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class EMAContainer:
    """Container for multiple EMA instances (with different decay) of a model."""

    def __init__(self, model, train_decays=[0.9999], force_cpu=False):
        self.model = model
        self.train_decays = train_decays
        self.force_cpu = force_cpu
        self.ema_models = {
            decay: ModelEmaV2(
                model,
                decay=decay,
                device="cpu" if force_cpu else None,
            )
            for decay in self.train_decays
        }
        logger.warning(
            "Created %d ema instances with decays=%s",
            len(self.train_decays),
            self.train_decays,
        )

    def state_dict(self):
        return {k: v.module.state_dict() for k, v in self.ema_models.items()}

    def to(self, device):
        logger.warning("EMAContainer.to(device) is experimental!")
        for ema_model in self.ema_models.values():
            ema_model.to(device)

    def update(self, model):
        for ema_model in self.ema_models.values():
            ema_model.update(model)

    def get_models(
        self, decays: Optional[List[float]] = None
    ) -> List[Tuple[str, nn.Module]]:
        decays = decays or self.train_decays
        return {decay: self.ema_models[decay].module for decay in decays}

    def get_model(self, decay):
        return self.ema_models[decay].module

    def update_dist_bn(self, world_size, dist_bn_method="reduce"):
        # update EMA dist BN
        # https://github.com/huggingface/pytorch-image-models/blob/main/train.py#L801
        assert world_size > 1
        for ema_model in self.ema_models.values():
            if not self.force_cpu:
                if dist_bn_method in ("broadcast", "reduce"):
                    logger.warning(
                        "Distributed BN update with world_size=%d, method=%s",
                        world_size,
                        dist_bn_method,
                    )
                    distribute_bn(
                        ema_model, world_size, reduce=(dist_bn_method == "reduce")
                    )
                else:
                    raise ValueError