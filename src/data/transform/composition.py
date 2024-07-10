from __future__ import division

import random
import typing
import warnings
from collections import defaultdict
from typing import Tuple

import numpy as np

from src.data.transform import utils
from src.data.transform.interface import BasicTransform

from .serialization import (SERIALIZABLE_REGISTRY, Serializable,
                            get_shortest_class_fullname,
                            instantiate_nonserializable)

__all__ = [
    "BaseCompose",
    "Compose",
    "SomeOf",
    "OneOf",
    "OneOrOther",
    "ReplayCompose",
    "Sequential",
]

REPR_INDENT_STEP = 4
TransformType = typing.Union[BasicTransform, "BaseCompose"]
TransformsSeqType = typing.Sequence[TransformType]


def get_always_apply(
    transforms: typing.Union["BaseCompose", TransformsSeqType]
) -> TransformsSeqType:
    new_transforms: typing.List[TransformType] = []
    for transform in transforms:  # type: ignore
        if isinstance(transform, BaseCompose):
            new_transforms.extend(get_always_apply(transform))
        elif transform.always_apply:
            new_transforms.append(transform)
    return new_transforms


class BaseCompose(Serializable):

    def __init__(self, transforms: TransformsSeqType, p: float):
        if isinstance(transforms, (BaseCompose, BasicTransform)):
            warnings.warn(
                "transforms is single transform, but a sequence is expected! Transform will be wrapped into list."
            )
            transforms = [transforms]

        self.transforms = transforms
        self.p = p

        self.replay_mode = False
        self.applied_in_replay = False
        self._validate_transforms()

    def _validate_transforms(self):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.transforms)

    def __call__(self, *args, **data) -> typing.Dict[str, typing.Any]:
        raise NotImplementedError

    def __getitem__(self, item: int) -> TransformType:  # type: ignore
        return self.transforms[item]

    def __repr__(self) -> str:
        return self.indented_repr()

    def indented_repr(self, indent: int = REPR_INDENT_STEP) -> str:
        args = {
            k: v
            for k, v in self._to_dict().items()
            if not (k.startswith("__") or k == "transforms")
        }
        repr_string = self.__class__.__name__ + "(["
        for t in self.transforms:
            repr_string += "\n"
            if hasattr(t, "indented_repr"):
                t_repr = t.indented_repr(indent +
                                         REPR_INDENT_STEP)  # type: ignore
            else:
                t_repr = repr(t)
            repr_string += " " * indent + t_repr + ","
        repr_string += "\n" + " " * (indent -
                                     REPR_INDENT_STEP) + "], {args})".format(
                                         args=utils.format_args(args))
        return repr_string

    @classmethod
    def get_class_fullname(cls) -> str:
        return get_shortest_class_fullname(cls)

    @classmethod
    def is_serializable(cls) -> bool:
        return True

    def _to_dict(self) -> typing.Dict[str, typing.Any]:
        return {
            "__class_fullname__": self.get_class_fullname(),
            "p": self.p,
            "transforms":
            [t._to_dict() for t in self.transforms],  # skipcq: PYL-W0212
        }

    def get_dict_with_id(self) -> typing.Dict[str, typing.Any]:
        return {
            "__class_fullname__": self.get_class_fullname(),
            "id": id(self),
            "params": None,
            "transforms": [t.get_dict_with_id() for t in self.transforms],
        }

    def add_targets(
            self,
            additional_targets: typing.Optional[typing.Dict[str,
                                                            str]]) -> None:
        if additional_targets:
            for t in self.transforms:
                t.add_targets(additional_targets)

    def set_deterministic(self, flag: bool, save_key: str = "replay") -> None:
        for t in self.transforms:
            t.set_deterministic(flag, save_key)


class Compose(BaseCompose):
    """Compose transforms and handle all transformations regarding bounding boxes

    Args:
        transforms (list): list of transformations to compose.
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}
        p (float): probability of applying all list of transforms. Default: 1.0.
        is_check_shapes (bool): If True shapes consistency of images/mask/masks would be checked on each call. If you
            would like to disable this check - pass False (do it only if you are sure in your data consistency).
    """

    def __init__(
        self,
        transforms: TransformsSeqType,
        additional_targets: typing.Optional[typing.Dict[str, str]] = None,
        p: float = 1.0,
    ):
        super(Compose, self).__init__(transforms, p)

        if additional_targets is None:
            additional_targets = {}

        self.additional_targets = additional_targets

        self.add_targets(additional_targets)

    def _validate_transforms(self):
        pass

    def __call__(self,
                 *args,
                 force_apply: bool = False,
                 **data) -> typing.Dict[str, typing.Any]:
        if args:
            raise KeyError(
                "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            )

        assert isinstance(
            force_apply, (bool, int)), "force_apply must have bool or int type"
        need_to_run = force_apply or random.random() < self.p
        transforms = self.transforms if need_to_run else get_always_apply(
            self.transforms)

        for _idx, t in enumerate(transforms):
            data = t(**data)
        data = Compose._make_targets_contiguous(
            data)  # ensure output targets are contiguous
        return data

    def _to_dict(self) -> typing.Dict[str, typing.Any]:
        dictionary = super(Compose, self)._to_dict()
        dictionary.update({
            "additional_targets": self.additional_targets,
        })
        return dictionary

    def get_dict_with_id(self) -> typing.Dict[str, typing.Any]:
        dictionary = super().get_dict_with_id()
        dictionary.update({
            "additional_targets": self.additional_targets,
            "params": None,
        })
        return dictionary

    @staticmethod
    def _make_targets_contiguous(
            data: typing.Dict[str,
                              typing.Any]) -> typing.Dict[str, typing.Any]:
        result = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
            result[key] = value
        return result


class OneOf(BaseCompose):
    """Select one of transforms to apply. Selected transform will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 0.5.
    """

    def __init__(self, transforms: TransformsSeqType, p: float = 0.5):
        super(OneOf, self).__init__(transforms, p)
        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    @property
    def input_shape(self) -> Tuple:
        return self.transforms[0].input_shape

    @property
    def output_shape(self) -> Tuple:
        return self.transforms[0].output_shape

    def _validate_transforms(self):
        pass

    def __call__(self,
                 *args,
                 force_apply: bool = False,
                 **data) -> typing.Dict[str, typing.Any]:
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
            return data

        if self.transforms_ps and (force_apply or random.random() < self.p):
            idx: int = utils.choice(len(self.transforms), p=self.transforms_ps)
            t = self.transforms[idx]
            data = t(force_apply=True, **data)
        return data


class SomeOf(BaseCompose):
    """Select N transforms to apply. Selected transforms will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

    Args:
        transforms (list): list of transformations to compose.
        n (int): number of transforms to apply.
        replace (bool): Whether the sampled transforms are with or without replacement. Default: True.
        p (float): probability of applying selected transform. Default: 1.
    """

    def __init__(self,
                 transforms: TransformsSeqType,
                 n: int,
                 replace: bool = True,
                 p: float = 1):
        super(SomeOf, self).__init__(transforms, p)
        self.n = n
        self.replace = replace
        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def _validate_transforms(self):
        pass

    def __call__(self,
                 *args,
                 force_apply: bool = False,
                 **data) -> typing.Dict[str, typing.Any]:
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
            return data

        if self.transforms_ps and (force_apply or random.random() < self.p):
            idx = utils.choice(len(self.transforms),
                               size=self.n,
                               replace=self.replace,
                               p=self.transforms_ps)
            for i in idx:  # type: ignore
                t = self.transforms[i]
                data = t(force_apply=True, **data)
        return data

    def _to_dict(self) -> typing.Dict[str, typing.Any]:
        dictionary = super(SomeOf, self)._to_dict()
        dictionary.update({"n": self.n, "replace": self.replace})
        return dictionary


class OneOrOther(BaseCompose):
    """Select one or another transform to apply. Selected transform will be called with `force_apply=True`."""

    def __init__(
        self,
        first: typing.Optional[TransformType] = None,
        second: typing.Optional[TransformType] = None,
        transforms: typing.Optional[TransformsSeqType] = None,
        p: float = 0.5,
    ):
        if transforms is None:
            if first is None or second is None:
                raise ValueError(
                    "You must set both first and second or set transforms argument."
                )
            transforms = [first, second]
        super(OneOrOther, self).__init__(transforms, p)
        if len(self.transforms) != 2:
            warnings.warn("Length of transforms is not equal to 2.")

    def _validate_transforms(self):
        pass

    def __call__(self,
                 *args,
                 force_apply: bool = False,
                 **data) -> typing.Dict[str, typing.Any]:
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
            return data

        if random.random() < self.p:
            return self.transforms[0](force_apply=True, **data)

        return self.transforms[-1](force_apply=True, **data)


class ReplayCompose(Compose):

    def __init__(
        self,
        transforms: TransformsSeqType,
        additional_targets: typing.Optional[typing.Dict[str, str]] = None,
        p: float = 1.0,
        save_key: str = "replay",
    ):
        super(ReplayCompose, self).__init__(transforms, additional_targets, p)
        self.set_deterministic(True, save_key=save_key)
        self.save_key = save_key

    def _validate_transforms(self):
        pass

    def __call__(self,
                 *args,
                 force_apply: bool = False,
                 **kwargs) -> typing.Dict[str, typing.Any]:
        kwargs[self.save_key] = defaultdict(dict)
        result = super(ReplayCompose, self).__call__(force_apply=force_apply,
                                                     **kwargs)
        serialized = self.get_dict_with_id()
        self.fill_with_params(serialized, result[self.save_key])
        self.fill_applied(serialized)
        result[self.save_key] = serialized
        return result

    @staticmethod
    def replay(saved_augmentations: typing.Dict[str, typing.Any],
               **kwargs) -> typing.Dict[str, typing.Any]:
        augs = ReplayCompose._restore_for_replay(saved_augmentations)
        return augs(force_apply=True, **kwargs)

    @staticmethod
    def _restore_for_replay(
            transform_dict: typing.Dict[str, typing.Any],
            lambda_transforms: typing.Optional[dict] = None) -> TransformType:
        """
        Args:
            lambda_transforms (dict): A dictionary that contains lambda transforms, that
            is instances of the Lambda class.
                This dictionary is required when you are restoring a pipeline that contains lambda transforms. Keys
                in that dictionary should be named same as `name` arguments in respective lambda transforms from
                a serialized pipeline.
        """
        applied = transform_dict["applied"]
        params = transform_dict["params"]
        lmbd = instantiate_nonserializable(transform_dict, lambda_transforms)
        if lmbd:
            transform = lmbd
        else:
            name = transform_dict["__class_fullname__"]
            args = {
                k: v
                for k, v in transform_dict.items()
                if k not in ["__class_fullname__", "applied", "params"]
            }
            cls = SERIALIZABLE_REGISTRY[name]
            if "transforms" in args:
                args["transforms"] = [
                    ReplayCompose._restore_for_replay(
                        t, lambda_transforms=lambda_transforms)
                    for t in args["transforms"]
                ]
            transform = cls(**args)

        transform = typing.cast(BasicTransform, transform)
        if isinstance(transform, BasicTransform):
            transform.params = params
        transform.replay_mode = True
        transform.applied_in_replay = applied
        return transform

    def fill_with_params(self, serialized: dict, all_params: dict) -> None:
        params = all_params.get(serialized.get("id"))
        serialized["params"] = params
        del serialized["id"]
        for transform in serialized.get("transforms", []):
            self.fill_with_params(transform, all_params)

    def fill_applied(self, serialized: typing.Dict[str, typing.Any]) -> bool:
        if "transforms" in serialized:
            applied = [self.fill_applied(t) for t in serialized["transforms"]]
            serialized["applied"] = any(applied)
        else:
            serialized["applied"] = serialized.get("params") is not None
        return serialized["applied"]

    def _to_dict(self) -> typing.Dict[str, typing.Any]:
        dictionary = super(ReplayCompose, self)._to_dict()
        dictionary.update({"save_key": self.save_key})
        return dictionary


class Sequential(BaseCompose):
    """Sequentially applies all transforms to targets.

    Note:
        This transform is not intended to be a replacement for `Compose`. Instead, it should be used inside `Compose`
        the same way `OneOf` or `OneOrOther` are used. For instance, you can combine `OneOf` with `Sequential` to
        create an augmentation pipeline that contains multiple sequences of augmentations and applies one randomly
        chose sequence to input data (see the `Example` section for an example definition of such pipeline).

    Example:
        >>> import albumentations as A
        >>> transform = A.Compose([
        >>>    A.OneOf([
        >>>        A.Sequential([
        >>>            A.HorizontalFlip(p=0.5),
        >>>            A.ShiftScaleRotate(p=0.5),
        >>>        ]),
        >>>        A.Sequential([
        >>>            A.VerticalFlip(p=0.5),
        >>>            A.RandomBrightnessContrast(p=0.5),
        >>>        ]),
        >>>    ], p=1)
        >>> ])
    """

    def _validate_transforms(self):
        pass

    def __init__(self, transforms: TransformsSeqType, p: float = 0.5):
        super().__init__(transforms, p)

    def __call__(self, *args, **data) -> typing.Dict[str, typing.Any]:
        for t in self.transforms:
            data = t(**data)
        return data
