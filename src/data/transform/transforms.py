import logging
import math
import random
import typing
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d

from src.data.transform import functional as F
from src.data.transform import utils
from src.data.transform.composition import BaseCompose
from src.data.transform.interface import (
    BasicTransform,
    DataTransform,
    ScaleFloatType,
    to_tuple,
)

logger = logging.getLogger(__name__)

TransformType = typing.Union[BasicTransform, BaseCompose]
REPR_INDENT_STEP = 4

__all__ = []


class _Template(DataTransform):

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

    def apply(self, data, **params):
        pass

    def get_params(self) -> Dict:
        return super().get_params()

    @property
    def targets_as_params(self) -> List[str]:
        return []

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def get_transform_init_args_names(self):
        return tuple()
