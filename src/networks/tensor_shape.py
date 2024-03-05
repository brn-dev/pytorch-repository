from dataclasses import dataclass
from typing import Optional, Literal

import sympy as sp


class TensorShape:

    @dataclass
    class DimensionInfo:
        symbol: sp.Symbol
        size: sp.Expr

        def copy(self) -> 'TensorShape.DimensionInfo':
            return TensorShape.DimensionInfo(symbol=self.symbol, size=self.size)

    FEATURES_KEY = 'features'

    BATCH_KEY = 'batch'
    DimKeyType = Literal['features', 'batch'] | str

    dimensions: dict[DimKeyType, DimensionInfo]

    def __init__(
            self,
            features: sp.Expr | int | None = None,
            batch: sp.Expr | int | None = None,
            **additional_dimensions: sp.Expr | int | None,
    ):
        self.dimensions = {}
        self[self.FEATURES_KEY] = features
        self[self.BATCH_KEY] = batch
        for dim_key, dim_value in additional_dimensions.items():
            self[dim_key] = dim_value

    def __getitem__(self, dim_key: DimKeyType) -> sp.core.Expr:
        return self.dimensions[dim_key].size

    def __setitem__(self, dim_key: DimKeyType, value: sp.Expr | int | None):
        symbol = sp.symbols(dim_key)
        if value is None:
            value = symbol
        elif not isinstance(value, sp.Basic):
            value = sp.sympify(value)
        self.dimensions[dim_key] = TensorShape.DimensionInfo(symbol, value)

    @property
    def completely_definite(self) -> bool:
        return all(self.is_definite(dim_key) for dim_key in self.dimensions.keys())

    @property
    def definite_symbols(self) -> set[sp.Symbol]:
        return set(
            dim_info.symbol
            for dim_key, dim_info
            in self.dimensions.items()
            if self.is_definite(dim_key)
        )

    def is_definite(self, dim_key: DimKeyType) -> bool:
        return not self[dim_key].free_symbols

    def definite(self, dim_key: DimKeyType) -> tuple[bool, Optional[int]]:
        dim_shape = self[dim_key]

        is_definite = self.is_definite(dim_key)
        if is_definite:
            value = int(dim_shape.evalf())
        else:
            value = None
        return is_definite, value

    def rename_dims(self, dim_keys: dict[DimKeyType, DimKeyType]) -> 'TensorShape':
        subs = {
            sp.symbols(from_key): sp.symbols(to_key)
            for from_key, to_key
            in dim_keys.items()
        }

        result_shape = self.copy()

        for dim_info in result_shape.dimensions.values():
            dim_info.size = dim_info.size.subs(subs)
        for from_key, to_key in dim_keys.items():
            result_shape[to_key] = result_shape[from_key]
            if from_key != to_key:
                del result_shape.dimensions[from_key]

        return result_shape


    def evaluate_forward(self, in_shape: 'TensorShape') -> 'TensorShape':
        result_shape = self.copy()

        for dim_key in self.dimensions.keys():
            assert not self.is_definite(dim_key)

            dim_size = self.dimensions[dim_key].size
            free_symbols = dim_size.free_symbols

            end_size = dim_size.evalf(subs={
                free_symbol: in_shape_dim
                for free_symbol
                in free_symbols
                if (in_shape_dim := in_shape[str(free_symbol)]) != free_symbol
            })

            if not end_size.free_symbols and end_size % 1.0 != 0.0:
                raise ValueError(f'Forward evaluation resulted in {end_size = }, should be integer')

            result_shape[dim_key] = end_size

        return result_shape

    def evaluate_backward(self, out_shape: 'TensorShape') -> 'TensorShape':
        result_shape = self.copy()

        for dim_key in self.dimensions.keys():
            assert not self.is_definite(dim_key)

            dim_info = self.dimensions[dim_key]
            dim_symbol = dim_info.symbol
            dim_size = dim_info.size

            if len(dim_size.free_symbols) > 1:
                raise NotImplementedError('Backward shape evaluation with more than one free symbol is currently '
                                          'not implemented')

            out_shape_dim_size = out_shape[str(dim_symbol)]
            if out_shape_dim_size != dim_symbol:
                in_size = sp.solve(dim_size - out_shape_dim_size, dim_symbol)[-1].evalf()
            else:
                in_size = dim_size

            if not in_size.free_symbols and in_size % 1.0 != 0.0:
                raise ValueError(f'Backward evaluation resulted in {in_size = }, should be integer')

            result_shape[dim_key] = in_size

        return result_shape

    def copy(self) -> 'TensorShape':
        return TensorShape(**{
            dim_key: dim_info.size
            for dim_key, dim_info
            in self.dimensions.items()
        })
