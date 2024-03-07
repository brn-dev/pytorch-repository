from dataclasses import dataclass
from typing import Optional, Literal, Callable

import sympy as sp


class TensorShape:

    @dataclass
    class DimensionInfo:
        symbol: sp.Symbol
        size: sp.Expr

        def copy(self) -> 'TensorShape.DimensionInfo':
            return TensorShape.DimensionInfo(symbol=self.symbol, size=self.size)

    STRUCTURAL_PREFIX = 's_'

    FEATURES_KEY = 'features'
    BATCH_KEY = 'batch'
    DimKeyType = Literal['features', 'batch'] | str | sp.Symbol

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
        return self.dimensions[str(dim_key)].size

    def __setitem__(self, dim_key: DimKeyType, value: sp.Expr | int | None):
        dim_key = str(dim_key)
        symbol = TensorShape.to_symbol(dim_key)
        if value is None:
            value = symbol
        elif isinstance(value, int):
            value = sp.Integer(value)
        elif not isinstance(value, sp.Basic):
            value = sp.sympify(value)
        self.dimensions[dim_key] = TensorShape.DimensionInfo(symbol, value)

    def __contains__(self, item: DimKeyType):
        return str(item) in self.dimensions

    def __str__(self):
        return f'TensorShape(' \
               f'{", ".join([f"{str(symbol)} = {self[symbol]}" for symbol in self.symbols])}' \
               f')'

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def to_symbol(dim_key: DimKeyType) -> sp.Symbol:
        return sp.symbols(dim_key, integer=True)

    def create_dimension(self, dim_key: DimKeyType) -> sp.Expr:
        if dim_key in self:
            raise ValueError(f'Dimension {dim_key = } already exists')
        self[dim_key] = None
        return self[dim_key]

    def is_definite(self, dim_key: DimKeyType) -> bool:
        return not self[str(dim_key)].free_symbols

    @property
    def completely_definite(self) -> bool:
        return all(self.is_definite(dim_key) for dim_key in self.dimensions.keys())

    @property
    def symbols(self) -> set[sp.Symbol]:
        return self.symbols_matching(lambda _: True)

    @property
    def definite_symbols(self) -> set[sp.Symbol]:
        return self.symbols_matching(self.is_definite)

    @property
    def structural_symbols_ordered(self) -> list[sp.Symbol]:
        return list(sorted(
            self.symbols_matching(self.is_structural),
            key=lambda sym: str(sym)
        ))

    def symbols_matching(self, condition: Callable[[sp.Symbol], bool]) -> set[sp.Symbol]:
        return set(
            dim_info.symbol
            for dim_key, dim_info
            in self.dimensions.items()
            if condition(dim_info.symbol)
        )

    def definite_size(self, dim_key: DimKeyType) -> tuple[bool, Optional[int]]:
        dim_key = str(dim_key)
        dim_shape = self[dim_key]

        is_definite = self.is_definite(dim_key)
        if is_definite:
            value = int(dim_shape.evalf())
        else:
            value = None
        return is_definite, value

    def is_structural(self, dim_key: DimKeyType):
        return str(dim_key).startswith(self.STRUCTURAL_PREFIX)

    def create_structural_dimension(self, nr: int) -> tuple[str, sp.Expr]:
        dim_key = self.STRUCTURAL_PREFIX + str(nr)
        return dim_key, self.create_dimension(dim_key)

    def rename_dims(self, dim_keys: dict[DimKeyType, DimKeyType]) -> 'TensorShape':
        subs = {
            TensorShape.to_symbol(from_key): TensorShape.to_symbol(to_key)
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
        result_shape = in_shape.copy()

        for dim_key in self.dimensions.keys():

            dim_size = self.dimensions[dim_key].size
            free_symbols = dim_size.free_symbols

            if dim_key not in in_shape:
                result_shape[dim_key] = dim_size
                continue

            in_shape_symbols = in_shape.symbols
            for free_symbol in free_symbols:
                if free_symbol not in in_shape_symbols:
                    raise ValueError(f'Dimension size of "{dim_key}" has the following free symbols: {free_symbols} - '
                                     f'in_shape is missing "{free_symbol}"')

            end_size = dim_size.subs({
                free_symbol: in_shape[str(free_symbol)]
                for free_symbol
                in free_symbols
            }).evalf()

            if not end_size.free_symbols and end_size % 1.0 != 0.0:
                raise ValueError(f'Forward evaluation resulted in {end_size = }, should be integer')

            result_shape[dim_key] = end_size

        return result_shape

    def evaluate_backward(self, dim_key: DimKeyType, out_shape: 'TensorShape') -> sp.Expr:
        dim_key = str(dim_key)
        dim_info = self.dimensions[dim_key]
        dim_symbol = dim_info.symbol
        dim_size = dim_info.size

        if len(dim_size.free_symbols) > 1:
            raise NotImplementedError('Backward shape evaluation with more than one free symbol is currently '
                                      'not implemented')
        assert dim_symbol in dim_size.free_symbols

        out_shape_dim_size = out_shape[dim_symbol]
        if out_shape_dim_size != dim_symbol:
            in_size = sp.solve(dim_size - int(out_shape_dim_size), dim_symbol)[-1].evalf()
        else:
            in_size = dim_size

        if not in_size.free_symbols and in_size % 1.0 != 0.0:
            raise ValueError(f'Backward evaluation resulted in {in_size = }, should be integer')

        return in_size

    def copy(self) -> 'TensorShape':
        return TensorShape(**{
            dim_key: dim_info.size
            for dim_key, dim_info
            in self.dimensions.items()
        })
