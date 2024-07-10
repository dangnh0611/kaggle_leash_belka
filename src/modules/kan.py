# @TODO: default update_grid = False, support update_grid = True in KAN.forward() as in https://github.com/Blealtan/efficient-kan
# @TODO: EfficientKAN L1 regularization: KAN.regularization_loss() as in https://github.com/Blealtan/efficient-kan
# @TODO: Support official implementation in pykan: https://github.com/KindXiaoming/pykan

import torch
import torch.nn.functional as F
import math
from torch import nn
import torch
from torch.nn import functional as F

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init
import math
from functools import partial
import math
from src.modules.misc import make_divisible, get_act_fn
from torchvision.ops import StochasticDepth
from typing import Callable, Optional, List
import torch
import torch as th
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EfficientKANLayer(torch.nn.Module):
    """
    Ref: https://github.com/Blealtan/efficient-kan
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h +
             grid_range[0]).expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight,
                                       a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features,
                                 self.out_features) - 1 / 2) *
                     self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline
                 else 1.0) * self.curve2coeff(
                     self.grid.T[self.spline_order:-self.spline_order],
                     noise,
                 ))
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler,
                                               a=math.sqrt(5) *
                                               self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid)  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k + 1)]) /
                (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]) + (
                    (grid[:, k + 1:] - x) /
                    (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1)  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(
            -1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines,
                                            orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2)  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[torch.linspace(0,
                                                batch - 1,
                                                self.grid_size + 1,
                                                dtype=torch.int64,
                                                device=x.device)]

        uniform_step = (x_sorted[-1] - x_sorted[0] +
                        2 * margin) / self.grid_size
        grid_uniform = (torch.arange(
            self.grid_size + 1, dtype=torch.float32,
            device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] -
                        margin)

        grid = self.grid_eps * grid_uniform + (1 -
                                               self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1] - uniform_step * torch.arange(
                    self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(
                    1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(
            self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self,
                            regularize_activation=1.0,
                            regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (regularize_activation * regularization_loss_activation +
                regularize_entropy * regularization_loss_entropy)


class ChebyKANLayer(nn.Module):
    """
    This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
    Ref: https://github.com/SynodicMonth/ChebyKAN
    """

    def __init__(self, input_dim, output_dim, degree=4):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs,
                        mean=0.0,
                        std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1,
            self.degree + 1)  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum("bid,iod->bo", x,
                         self.cheby_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y


class FourierKANLayer(th.nn.Module):
    """
    https://github.com/GistNoesis/FourierKAN
    This is inspired by Kolmogorov-Arnold Networks but using 1d fourier coefficients 
    instead of splines coefficients. It should be easier to optimize as fourier are 
    more dense than spline (global vs local) Once convergence is reached you can 
    replace the 1d function with spline approximation for faster evaluation giving 
    almost the same result. The other advantage of using fourier over spline is that
    the function are periodic, and therefore more numerically bounded
    Avoiding the issues of going out of grid
    """

    def __init__(self,
                 inputdim,
                 outdim,
                 gridsize,
                 addbias=True,
                 smooth_initialization=False):
        super().__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        # With smooth_initialization, fourier coefficients are attenuated by the square of their frequency.
        # This makes KAN's scalar functions smooth at initialization.
        # Without smooth_initialization, high gridsizes will lead to high-frequency scalar functions,
        # with high derivatives and low correlation between similar inputs.
        grid_norm_factor = (
            th.arange(gridsize) +
            1)**2 if smooth_initialization else np.sqrt(gridsize)

        #The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        #then each coordinates of the output is of unit variance
        #independently of the various sizes
        self.fouriercoeffs = th.nn.Parameter(
            th.randn(2, outdim, inputdim, gridsize) /
            (np.sqrt(inputdim) * grid_norm_factor))
        if (self.addbias):
            self.bias = th.nn.Parameter(th.zeros(1, outdim))

    #x.shape ( ... , indim )
    #out.shape ( ..., outdim)
    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim, )
        x = th.reshape(x, (-1, self.inputdim))
        #Starting at 1 because constant terms are in the bias
        k = th.reshape(th.arange(1, self.gridsize + 1, device=x.device),
                       (1, 1, 1, self.gridsize))
        xrshp = th.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        #This should be fused to avoid materializing memory
        c = th.cos(k * xrshp)
        s = th.sin(k * xrshp)
        #We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them
        y = th.sum(c * self.fouriercoeffs[0:1], (-2, -1))
        y += th.sum(s * self.fouriercoeffs[1:2], (-2, -1))
        if (self.addbias):
            y += self.bias
        #End fuse
        '''
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = th.reshape(c,(1,x.shape[0],x.shape[1],self.gridsize))
        s = th.reshape(s,(1,x.shape[0],x.shape[1],self.gridsize))
        y2 = th.einsum( "dbik,djik->bj", th.concat([c,s],axis=0) ,self.fouriercoeffs)
        if( self.addbias):
            y2 += self.bias
        diff = th.sum((y2-y)**2)
        print("diff")
        print(diff) #should be ~0
        '''
        y = th.reshape(y, outshape)
        return y


class KAN(torch.nn.Sequential):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: List[int],
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                 kan_type='cheby',
                 dropout: float = 0.0,
                 last_norm=True,
                 last_dropout=True,
                 last_linear=False,
                 first_linear=False,
                 grid: int = 5,
                 k: int = 3,
                 **kwargs):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py

        if kan_type == 'cheby' and norm_layer is None:
            logger.warn(
                'Note: Since Chebyshev polynomials are defined on the interval [-1, 1], we need to use tanh to keep the input in that range. We also use LayerNorm to avoid gradient vanishing caused by tanh. Removing LayerNorm will cause the network really hard to train. More: https://github.com/SynodicMonth/ChebyKAN'
            )

        layers = []
        in_dim = in_channels

        for i, hidden_dim in enumerate(hidden_channels):
            is_last = (i == len(hidden_channels) - 1)
            is_not_last = not is_last
            # https://github.com/KindXiaoming/pykan?tab=readme-ov-file#authors-note
            # GraphKAN (https://github.com/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks)
            # suggests that KANs should better be used in latent space
            # (need embedding and unembedding linear layers after inputs and before outputs)
            if (i == 0 and first_linear) or (is_last and last_linear):
                kan_layer = nn.Linear(in_dim, hidden_dim, bias=True)
            elif kan_type == 'efficient':
                kan_layer = EfficientKANLayer(
                    in_dim,
                    hidden_dim,
                    grid,
                    k,
                    scale_noise=kwargs.get('scale_noise', 0.1),
                    scale_base=kwargs.get('scale_base', 1),
                    scale_spline=kwargs.get('scale_spline', 1),
                    enable_standalone_scale_spline=kwargs.get(
                        'enable_standalone_scale_spline', True),
                    base_activation=get_act_fn(
                        kwargs.get('base_activation', 'silu')),
                    grid_eps=kwargs.get('gride_eps', 0.02),
                    grid_range=kwargs.get('grid_range', [-1, 1]))
            elif kan_type == 'cheby':
                kan_layer = ChebyKANLayer(in_dim, hidden_dim, k)
            elif kan_type == 'fourier':
                kan_layer = FourierKANLayer(
                    in_dim,
                    hidden_dim,
                    grid,
                    addbias=kwargs.get('bias', True),
                    smooth_initialization=kwargs.get('smooth_initialization',
                                                     False),
                )
            elif kan_type == 'naive':
                raise NotImplementedError
            else:
                raise ValueError
            layers.append(kan_layer)
            if norm_layer is not None and (is_not_last or last_norm):
                layers.append(norm_layer(hidden_dim))
            if is_not_last or last_dropout:
                layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        super().__init__(*layers)
