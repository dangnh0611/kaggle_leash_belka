from functools import partial
from torch import nn


def get_act_fn(name, inplace=False, **kwargs):
    _ACTS = {
        "gelu": partial(nn.GELU, **kwargs),
        "relu": partial(nn.ReLU, inplace=inplace, **kwargs),
        "relu6": partial(nn.ReLU6, inplace=inplace, **kwargs),
        "silu": partial(nn.SiLU, inplace=inplace, **kwargs),  # Swish
        "selu": partial(nn.SELU, inplace=inplace, **kwargs),
        "hs": partial(nn.Hardswish, inplace=inplace, **kwargs),
        "sigmoid": partial(nn.Sigmoid, **kwargs),
        "tanh": partial(nn.Tanh, **kwargs),
        "mish": partial(nn.Mish, inplace=inplace, **kwargs),
        "leakyrelu": partial(nn.LeakyReLU, inplace=inplace, **kwargs),
        "prelu": partial(nn.PReLU, **kwargs),
        "elu": partial(nn.ELU, inplace=inplace, **kwargs),
        "glu": partial(nn.GLU),
    }
    return _ACTS[name]


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def make_net_dims(depth,
                  base_dim,
                  dim_scale_method,
                  width_multiplier=1,
                  divisor=8):
    ret_dims = [base_dim]
    if "add" in dim_scale_method:
        _method, coef = dim_scale_method.split("_")
        assert _method == "add"
        coef = float(coef)
        for _ in range(depth):
            ret_dims.append(make_divisible(ret_dims[-1] + coef, divisor))
    elif "mul" in dim_scale_method:
        _method, coef = dim_scale_method.split("_")
        assert _method == "mul"
        coef = float(coef)
        for _ in range(depth):
            ret_dims.append(make_divisible(int(ret_dims[-1] * coef), divisor))
    else:
        raise ValueError
    ret_dims = [
        make_divisible(e * width_multiplier, divisor) for e in ret_dims
    ]
    return ret_dims[1:]


