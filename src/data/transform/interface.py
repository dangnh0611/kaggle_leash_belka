from __future__ import absolute_import

import random
from copy import deepcopy
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple,
                    Union, cast)
from warnings import warn

import numpy as np

from src.data.transform.serialization import (Serializable,
                                              get_shortest_class_fullname)
from src.data.transform import utils

NumType = Union[int, float, np.ndarray]
BoxInternalType = Tuple[float, float, float, float]
BoxType = Union[BoxInternalType, Tuple[float, float, float, float, Any]]
KeypointInternalType = Tuple[float, float, float, float]
KeypointType = Union[KeypointInternalType, Tuple[float, float, float, float,
                                                 Any]]
ImageColorType = Union[float, Sequence[float]]

ScaleFloatType = Union[float, Tuple[float, float]]
ScaleIntType = Union[int, Tuple[int, int]]

FillValueType = Optional[Union[int, float, Sequence[int], Sequence[float]]]


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple
    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, Sequence):
        if len(param) != 2:
            raise ValueError("to_tuple expects 1 or 2 values")
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)

class BasicTransform(Serializable):
    call_backup = None
    interpolation: Any
    fill_value: Any
    mask_fill_value: Any

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.p = p
        self.always_apply = always_apply
        self._additional_targets: Dict[str, str] = {}

        # replay mode params
        self.deterministic = False
        self.save_key = "replay"
        self.params: Dict[Any, Any] = {}
        self.replay_mode = False
        self.applied_in_replay = False

    def __call__(self,
                 *args,
                 force_apply: bool = False,
                 **kwargs) -> Dict[str, Any]:
        if args:
            raise KeyError(
                "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            )
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if (random.random() < self.p) or self.always_apply or force_apply:
            params = self.get_params()

            if self.targets_as_params:
                assert all(key in kwargs for key in
                           self.targets_as_params), "{} requires {}".format(
                               self.__class__.__name__, self.targets_as_params)
                targets_as_params = {
                    k: kwargs[k]
                    for k in self.targets_as_params
                }
                params_dependent_on_targets = self.get_params_dependent_on_targets(
                    targets_as_params)
                params.update(params_dependent_on_targets)
            if self.deterministic:
                if self.targets_as_params:
                    warn(
                        self.get_class_fullname() +
                        " could work incorrectly in ReplayMode for other input data"
                        " because its' params depend on targets.")
                kwargs[self.save_key][id(self)] = deepcopy(params)
            return self.apply_with_params(params, **kwargs)

        return kwargs

    def apply_with_params(self, params: Dict[str, Any],
                          **kwargs) -> Dict[str, Any]:  # skipcq: PYL-W0613
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}
        for key, arg in kwargs.items():
            if arg is not None:
                target_function = self._get_target_function(key)
                target_validator = self._get_target_validator(key)
                target_dependencies = {
                    k: kwargs[k]
                    for k in self.target_dependence.get(key, [])
                }
                output = target_function(arg,
                                         **dict(params, **target_dependencies))
                target_validator(arg, output)
                res[key] = output

            else:
                res[key] = None
        return res

    def set_deterministic(self,
                          flag: bool,
                          save_key: str = "replay") -> "BasicTransform":
        assert save_key != "params", "params save_key is reserved"
        self.deterministic = flag
        self.save_key = save_key
        return self

    def __repr__(self) -> str:
        state = self.get_base_init_args()
        state.update(self.get_transform_init_args())
        return "{name}({args})".format(name=self.__class__.__name__,
                                       args=utils.format_args(state))

    def _get_target_function(self, key: str) -> Callable:
        transform_key = key
        if key in self._additional_targets:
            transform_key = self._additional_targets.get(key, key)

        target_function = self.targets.get(transform_key, lambda x, **p: x)
        return target_function

    def _get_target_validator(self, key: str) -> Callable:
        transform_key = key
        if key in self._additional_targets:
            transform_key = self._additional_targets.get(key, key)

        target_validator = self.target_validators.get(transform_key,
                                                      lambda x, y: None)
        return target_validator

    def apply(self, sequence: np.ndarray, **params) -> np.ndarray:
        raise NotImplementedError

    def get_params(self) -> Dict:
        return {}

    @property
    def targets(self) -> Dict[str, Callable]:
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError

    @property
    def target_validators(self) -> Dict[str, Callable]:
        raise NotImplementedError

    def update_params(self, params: Dict[str, Any],
                      **kwargs) -> Dict[str, Any]:
        return params

    @property
    def target_dependence(self) -> Dict:
        return {}

    def add_targets(self, additional_targets: Dict[str, str]):
        """Add targets to transform them the same way as one of existing targets
        ex: {'target_image': 'image'}
        ex: {'obj1_mask': 'mask', 'obj2_mask': 'mask'}
        by the way you must have at least one object with key 'image'

        Args:
            additional_targets (dict): keys - new target name, values - old target name. ex: {'image2': 'image'}
        """
        self._additional_targets = additional_targets

    @property
    def targets_as_params(self) -> List[str]:
        return []

    def get_params_dependent_on_targets(
            self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            "Method get_params_dependent_on_targets is not implemented in class "
            + self.__class__.__name__)

    @classmethod
    def get_class_fullname(cls) -> str:
        return get_shortest_class_fullname(cls)

    @classmethod
    def is_serializable(cls):
        return True

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        raise NotImplementedError(
            "Class {name} is not serializable because the `get_transform_init_args_names` method is not "
            "implemented".format(name=self.get_class_fullname()))

    def get_base_init_args(self) -> Dict[str, Any]:
        return {"always_apply": self.always_apply, "p": self.p}

    def get_transform_init_args(self) -> Dict[str, Any]:
        return {
            k: getattr(self, k)
            for k in self.get_transform_init_args_names()
        }

    def _to_dict(self) -> Dict[str, Any]:
        state = {"__class_fullname__": self.get_class_fullname()}
        state.update(self.get_base_init_args())
        state.update(self.get_transform_init_args())
        return state

    def get_dict_with_id(self) -> Dict[str, Any]:
        d = self._to_dict()
        d["id"] = id(self)
        return d


class DataTransform(BasicTransform):

    @property
    def target_validators(self) -> Dict[str, Callable[..., Any]]:
        return {}

    @property
    def targets(self) -> Dict[str, Callable]:
        return {"data": self.apply}
    
    def update_params(self, params: Dict[str, Any],
                      **kwargs) -> Dict[str, Any]:
        return params

    def apply(self, data: Dict, **params):
        raise NotImplementedError
