from typing import TypeVar, Union, Tuple, Any

import sympy as sp
import torch
from torch import nn, Tensor

from src.networks.core.net import Net
from src.networks.core.tensor_shape import TensorShape
from src.networks.tensor_operations.permute_dimensions import find_permutation
from src.torch_nn_modules import nn_pooling_classes, is_nn_convolutional_module, is_nn_pooling_module

T = TypeVar('T')
_scalar_or_tuple_1_t = Union[T, Tuple[T]]
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_scalar_or_tuple_3_t = Union[T, Tuple[T, T, T]]

_size_1_t = _scalar_or_tuple_1_t[int]
_size_2_t = _scalar_or_tuple_2_t[int]
_size_3_t = _scalar_or_tuple_3_t[int]

_int_size = Union[Tuple[int, ...], int]


def to_tuple(value: _int_size, length: int):
    if isinstance(value, tuple):
        return value
    return tuple(value for _ in range(length))


def compute_conv_in_out_shapes(
        conv: nn.Conv1d | nn.Conv2d | nn.Conv3d | Any  # ConvNd or PoolNd
):
    if is_nn_convolutional_module(conv):
        in_channels = conv.in_channels
        out_channels = conv.out_channels
    elif is_nn_pooling_module(conv):
        in_channels = sp.Symbol(TensorShape.FEATURES_KEY)
        out_channels = sp.Symbol(TensorShape.FEATURES_KEY)
    else:
        raise ValueError(f'Unknown type of conv argument ({type(conv)}')

    return _compute_conv_in_out_shapes(
        nr_dims=len(conv.kernel_size),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
    )


def _compute_conv_in_out_shapes(
        nr_dims: int,
        in_channels: int | sp.Expr,
        out_channels: int | sp.Expr,
        kernel_size: _int_size,
        stride: _int_size,
        padding: Union[_int_size, str],
        dilation: _int_size,
):
    in_shape, out_shape = TensorShape(features=in_channels), TensorShape(features=out_channels)

    if padding == 'same':
        for conv_dim_nr in range(nr_dims):
            in_shape.create_structural_dimension()
            out_shape.create_structural_dimension()
        return in_shape, out_shape
    if padding == 'valid':
        padding = 0

    kernel_size = to_tuple(kernel_size, length=nr_dims)
    stride = to_tuple(stride, length=nr_dims)
    padding = to_tuple(padding, length=nr_dims)
    dilation = to_tuple(dilation, length=nr_dims)

    for conv_dim_nr in range(nr_dims):
        dim_kernel_size = kernel_size[conv_dim_nr]
        dim_stride = stride[conv_dim_nr]
        dim_padding = padding[conv_dim_nr]
        dim_dilation = dilation[conv_dim_nr]

        dim_key, dim_symbol = out_shape.create_structural_dimension()
        out_shape[dim_key] = sp.floor(
            (dim_symbol + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) / dim_stride + 1
        )

        in_shape.create_structural_dimension()

    return in_shape, out_shape


# class Conv1dNet(Net, nn.Conv1d):
#
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             kernel_size: _size_1_t,
#             stride: _size_1_t = 1,
#             padding: Union[str, _size_1_t] = 0,
#             dilation: _size_1_t = 1,
#             groups: int = 1,
#             bias: bool = True,
#             padding_mode: str = 'zeros',
#             device=None,
#             dtype=None,
#             external_dim_order: list[str] = None,
#             internal_dim_order: list[str] = None,
#     ):
#         in_shape, out_shape = _compute_conv_in_out_shapes(
#             1, in_channels, out_channels, kernel_size, stride, padding, dilation
#         )
#         Net.__init__(self, in_shape=in_shape, out_shape=out_shape, allow_extra_dimensions=False)
#         nn.Conv1d.__init__(
#             self,
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias,
#             padding_mode=padding_mode,
#             device=device,
#             dtype=dtype,
#         )
#
#         structural_prefix = TensorShape.STRUCTURAL_PREFIX
#         if external_dim_order is None:
#             external_dim_order = [
#                 structural_prefix + str(0),
#                 'batch',
#                 'features',
#             ]
#         if internal_dim_order is None:
#             internal_dim_order = [
#                 'batch',
#                 'features',
#                 structural_prefix + str(0),
#             ]
#
#         self.enter_permutation = find_permutation(external_dim_order, internal_dim_order)
#         self.exit_permutation = find_permutation(internal_dim_order, external_dim_order)
#
#     def forward(self, x: Tensor) -> Tensor:
#         x = torch.permute(x, self.enter_permutation)
#         x = nn.Conv1d.forward(self, x)
#         x = torch.permute(x, self.exit_permutation)
#         return x
#
#
# class Conv2dNet(Net, nn.Conv2d):
#
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             kernel_size: _size_2_t,
#             stride: _size_2_t = 1,
#             padding: Union[str, _size_2_t] = 0,
#             dilation: _size_2_t = 1,
#             groups: int = 1,
#             bias: bool = True,
#             padding_mode: str = 'zeros',
#             device=None,
#             dtype=None,
#             external_dim_order: list[str] = None,
#             internal_dim_order: list[str] = None,
#     ):
#         in_shape, out_shape = _compute_conv_in_out_shapes(
#             2, in_channels, out_channels, kernel_size, stride, padding, dilation
#         )
#         Net.__init__(self, in_shape=in_shape, out_shape=out_shape, allow_extra_dimensions=False)
#         nn.Conv2d.__init__(
#             self,
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias,
#             padding_mode=padding_mode,
#             device=device,
#             dtype=dtype,
#         )
#
#         structural_prefix = TensorShape.STRUCTURAL_PREFIX
#         if external_dim_order is None:
#             external_dim_order = [
#                 structural_prefix + str(0),
#                 structural_prefix + str(1),
#                 'batch',
#                 'features',
#             ]
#         if internal_dim_order is None:
#             internal_dim_order = [
#                 'batch',
#                 'features',
#                 structural_prefix + str(0),
#                 structural_prefix + str(1),
#             ]
#
#         self.enter_permutation = find_permutation(external_dim_order, internal_dim_order)
#         self.exit_permutation = find_permutation(internal_dim_order, external_dim_order)
#
#     def forward(self, x: Tensor) -> Tensor:
#         x = torch.permute(x, self.enter_permutation)
#         x = nn.Conv2d.forward(self, x)
#         x = torch.permute(x, self.exit_permutation)
#         return x
#
#
# class Conv3dNet(Net, nn.Conv3d):
#
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             kernel_size: _size_3_t,
#             stride: _size_3_t = 1,
#             padding: Union[str, _size_3_t] = 0,
#             dilation: _size_3_t = 1,
#             groups: int = 1,
#             bias: bool = True,
#             padding_mode: str = 'zeros',
#             device=None,
#             dtype=None,
#             external_dim_order: list[str] = None,
#             internal_dim_order: list[str] = None,
#     ):
#         in_shape, out_shape = _compute_conv_in_out_shapes(
#             3, in_channels, out_channels, kernel_size, stride, padding, dilation
#         )
#         Net.__init__(self, in_shape=in_shape, out_shape=out_shape, allow_extra_dimensions=False)
#         nn.Conv3d.__init__(
#             self,
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias,
#             padding_mode=padding_mode,
#             device=device,
#             dtype=dtype,
#         )
#
#         structural_prefix = TensorShape.STRUCTURAL_PREFIX
#         if external_dim_order is None:
#             external_dim_order = [
#                 structural_prefix + str(0),
#                 structural_prefix + str(1),
#                 structural_prefix + str(2),
#                 'batch',
#                 'features',
#             ]
#         if internal_dim_order is None:
#             internal_dim_order = [
#                 'batch',
#                 'features',
#                 structural_prefix + str(0),
#                 structural_prefix + str(1),
#                 structural_prefix + str(2),
#             ]
#
#         self.enter_permutation = find_permutation(external_dim_order, internal_dim_order)
#         self.exit_permutation = find_permutation(internal_dim_order, external_dim_order)
#
#     def forward(self, x: Tensor) -> Tensor:
#         x = torch.permute(x, self.enter_permutation)
#         x = nn.Conv3d.forward(self, x)
#         x = torch.permute(x, self.exit_permutation)
#         return x
