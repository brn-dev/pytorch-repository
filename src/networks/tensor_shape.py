from dataclasses import dataclass
from typing import Optional, Literal

import sympy as sp


class TensorShape:

    @dataclass
    class DimensionInfo:
        symbol: sp.Symbol
        size: sp.Expr

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
        symbol = self.get_symbol(dim_key)
        if value is None:
            value = symbol
        elif not isinstance(value, sp.Basic):
            value = sp.sympify(value)
        self.dimensions[dim_key] = TensorShape.DimensionInfo(symbol, value)

    def get_symbol(self, dim_key: DimKeyType):
        dim_info = self.dimensions.get(dim_key, None)
        if dim_info is not None:
            return dim_info.symbol
        return sp.symbols(dim_key)

    def defined(self, dim_key: DimKeyType) -> tuple[bool, Optional[int]]:
        dim_shape = self[dim_key]

        is_defined = not dim_shape.free_symbols
        if is_defined:
            value = int(dim_shape.evalf())
        else:
            value = None
        return is_defined, value

    def evaluate(self, dim_key: DimKeyType, in_size: int):
        assert not self.defined(dim_key)[0]

        dim_info = self.dimensions[dim_key]
        dim_symbol = dim_info.symbol
        dim_size = dim_info.size

        out_size = dim_size.evalf(subs={dim_symbol: sp.Integer(in_size)})

        assert out_size % 1.0 == 0.0
        return int(out_size)

    def find_inverse(self, dim_key: DimKeyType, out_size: int) -> int:
        assert not self.defined(dim_key)[0]

        dim_info = self.dimensions[dim_key]
        dim_symbol = dim_info.symbol
        dim_size = dim_info.size

        in_size = sp.solve(dim_size - out_size, dim_symbol)[-1].evalf()

        assert in_size % 1.0 == 0.0
        return int(in_size)
