from typing import TypeVar, Union, Tuple

import sympy as sp
import torch
from torch import nn, Tensor

from src.networks.core.net import Net
from src.networks.core.tensor_shape import TensorShape
from src.networks.tensor_operations.functional import find_permutation

T = TypeVar('T')
_scalar_or_tuple_1_t = Union[T, Tuple[T]]
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_scalar_or_tuple_3_t = Union[T, Tuple[T, T, T]]

_size_1_t = _scalar_or_tuple_1_t[int]
_size_2_t = _scalar_or_tuple_2_t[int]
_size_3_t = _scalar_or_tuple_3_t[int]


def compute_conv_in_out_shapes(conv: nn.Conv1d | nn.Conv2d | nn.Conv3d):
    in_shape, out_shape = TensorShape(features=conv.in_channels), TensorShape(features=conv.out_channels)

    for conv_dim_nr in range(len(conv.kernel_size)):
        kernel_size = conv.kernel_size[conv_dim_nr]
        stride = conv.stride[conv_dim_nr]
        padding = conv.padding[conv_dim_nr]
        dilation = conv.dilation[conv_dim_nr]

        dim_key, dim_symbol = out_shape.create_structural_dimension()
        out_shape[dim_key] = sp.floor(
            (dim_symbol + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )

        in_shape.create_structural_dimension()

    return in_shape, out_shape


class Conv1dNet(Net, nn.Conv1d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_1_t,
            stride: _size_1_t = 1,
            padding: Union[str, _size_1_t] = 0,
            dilation: _size_1_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None,
            external_dim_order: list[str] = None,
            internal_dim_order: list[str] = None,
    ):
        nn.Conv1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.in_shape, self.out_shape = compute_conv_in_out_shapes(self)

        structural_prefix = TensorShape.STRUCTURAL_PREFIX
        if external_dim_order is None:
            external_dim_order = [
                structural_prefix + str(0),
                'batch',
                'features',
            ]
        if internal_dim_order is None:
            internal_dim_order = [
                'batch',
                'features',
                structural_prefix + str(0),
            ]

        self.enter_permutation = find_permutation(external_dim_order, internal_dim_order)
        self.exit_permutation = find_permutation(internal_dim_order, external_dim_order)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.permute(x, self.enter_permutation)
        x = nn.Conv1d.forward(self, x)
        x = torch.permute(x, self.exit_permutation)
        return x

class Conv2dNet(Net, nn.Conv2d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
            external_dim_order: list[str] = None,
            internal_dim_order: list[str] = None,
    ):
        nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.in_shape, self.out_shape = compute_conv_in_out_shapes(self)

        structural_prefix = TensorShape.STRUCTURAL_PREFIX
        if external_dim_order is None:
            external_dim_order = [
                structural_prefix + str(0),
                structural_prefix + str(1),
                'batch',
                'features',
            ]
        if internal_dim_order is None:
            internal_dim_order = [
                'batch',
                'features',
                structural_prefix + str(0),
                structural_prefix + str(1),
            ]

        self.enter_permutation = find_permutation(external_dim_order, internal_dim_order)
        self.exit_permutation = find_permutation(internal_dim_order, external_dim_order)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.permute(x, self.enter_permutation)
        x = nn.Conv2d.forward(self, x)
        x = torch.permute(x, self.exit_permutation)
        return x

class Conv3dNet(Net, nn.Conv3d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_3_t,
            stride: _size_3_t = 1,
            padding: Union[str, _size_3_t] = 0,
            dilation: _size_3_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
            external_dim_order: list[str] = None,
            internal_dim_order: list[str] = None,
    ):
        nn.Conv3d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.in_shape, self.out_shape = compute_conv_in_out_shapes(self)


        structural_prefix = TensorShape.STRUCTURAL_PREFIX
        if external_dim_order is None:
            external_dim_order = [
                structural_prefix + str(0),
                structural_prefix + str(1),
                structural_prefix + str(2),
                'batch',
                'features',
            ]
        if internal_dim_order is None:
            internal_dim_order = [
                'batch',
                'features',
                structural_prefix + str(0),
                structural_prefix + str(1),
                structural_prefix + str(2),
            ]

        self.enter_permutation = find_permutation(external_dim_order, internal_dim_order)
        self.exit_permutation = find_permutation(internal_dim_order, external_dim_order)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.permute(x, self.enter_permutation)
        x = nn.Conv3d.forward(self, x)
        x = torch.permute(x, self.exit_permutation)
        return x
