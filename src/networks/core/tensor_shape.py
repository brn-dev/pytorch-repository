import functools
from dataclasses import dataclass
from typing import Optional, Literal, Callable

import sympy as sp

DimKeyType = Literal['features', 'batch'] | str | sp.Symbol

class TensorShapeError(Exception):

    def __init__(self, message: str, parent_error: Exception = None, **shapes: 'TensorShape') -> None:
        self.message = message
        self.shapes = shapes
        self.parent_error = parent_error
        super().__init__(message, shapes, parent_error)

    def __str__(self):
        return f'{self.message} \n\n' + (''.join([f'\t {key}: {shape}\n' for key, shape in self.shapes.items()]))

    def __repr__(self):
        return self.__str__()


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

    REMOVED_DIM_VALUE = -1

    dimensions: dict[str, DimensionInfo]

    def __init__(
            self,
            features: sp.Expr | int | None = None,
            batch: sp.Expr | int | None = None,
            **auxiliary_dimensions: sp.Expr | int | None,
    ):
        self.dimensions = {}
        self[self.FEATURES_KEY] = features
        self[self.BATCH_KEY] = batch
        for dim_key, dim_value in auxiliary_dimensions.items():
            self[dim_key] = dim_value

    def __getitem__(self, dim_key: DimKeyType) -> sp.Expr:
        dim_key = _to_key(dim_key)
        return self.dimensions[dim_key].size

    def __setitem__(self, dim_key: DimKeyType, value: sp.Expr | int | None):
        dim_key = _to_key(dim_key)
        symbol = _to_symbol(dim_key)
        if value is None:
            value = symbol
        elif isinstance(value, int):
            value = sp.Integer(value)
        elif not isinstance(value, sp.Basic):
            value = sp.sympify(value)
        self.dimensions[dim_key] = TensorShape.DimensionInfo(symbol, value)

    def __delitem__(self, dim_key: DimKeyType):
        dim_key = _to_key(dim_key)
        del self.dimensions[dim_key]

    def __contains__(self, item: DimKeyType):
        return str(item) in self.dimensions

    def __str__(self):
        dim_names = [
            *self.structural_dimension_names,
            'batch',
            'features',
        ]

        auxiliary_dims = set(self.dimension_names) - set(dim_names)
        dim_names.extend(auxiliary_dims)

        dim_infos = [
            f"{dim} = {int(dim_size) if not (dim_size := self[dim]).free_symbols else str(dim_size)}"
            for dim
            in dim_names
        ]
        return f'TensorShape({", ".join(dim_infos)})'

    def __repr__(self):
        return self.__str__()

    def create_dimension(self, dim_key: DimKeyType) -> sp.Expr:
        dim_key = _to_key(dim_key)
        if dim_key in self:
            raise ValueError(f'Dimension {dim_key = } already exists')
        self[dim_key] = None
        return self[dim_key]

    def is_definite(self, dim_key: DimKeyType) -> bool:
        return not self[dim_key].free_symbols

    @property
    def completely_definite(self) -> bool:
        return all(self.is_definite(dim_key) for dim_key in self.dimensions.keys())

    @property
    def dimension_names(self) -> set[str]:
        return self.dimension_names_matching(lambda _: True)

    @property
    def definite_dimension_names(self) -> set[str]:
        return self.dimension_names_matching(self.is_definite)

    @property
    def structural_dimension_names(self) -> list[str]:
        return list(sorted(
            self.dimension_names_matching(self.is_structural),
            key=lambda sym: str(sym)
        ))

    def dimension_names_matching(self, condition: Callable[[DimKeyType], bool]) -> set[str]:
        return set(
            dim_key
            for dim_key
            in self.dimensions.keys()
            if condition(dim_key)
        )

    def try_get_definite_size(self, dim_key: DimKeyType) -> tuple[bool, Optional[int]]:
        dim_key = _to_key(dim_key)
        dim_shape = self[dim_key]

        is_definite = self.is_definite(dim_key)
        if is_definite:
            size = int(dim_shape.evalf())
        else:
            size = None
        return is_definite, size

    def get_definite_size(self, dim_key: DimKeyType) -> int:
        is_definite, size = self.try_get_definite_size(dim_key)
        if not is_definite:
            raise TensorShapeError(f'Dimension {dim_key} is not definite', _self=self)
        return size

    def is_structural(self, dim_key: DimKeyType):
        dim_key = _to_key(dim_key)
        return dim_key.startswith(self.STRUCTURAL_PREFIX)

    def create_structural_dimension(self, dim_nr: int | None = None) -> tuple[str, sp.Expr]:
        if dim_nr is None:
            dim_key = self.STRUCTURAL_PREFIX + str(len(self.structural_dimension_names))
        else:
            dim_key = self.STRUCTURAL_PREFIX + str(dim_nr)
        return dim_key, self.create_dimension(dim_key)

    def rename_dims(self, dim_keys: dict[DimKeyType, DimKeyType]) -> 'TensorShape':
        subs = {
            _to_symbol(from_key): _to_symbol(to_key)
            for from_key, to_key
            in dim_keys.items()
        }

        result_shape = self.copy()

        for dim_info in result_shape.dimensions.values():
            dim_info.size = dim_info.size.subs(subs)
        for from_key, to_key in dim_keys.items():
            result_shape[to_key] = result_shape[from_key]
            if from_key != to_key:
                del result_shape[from_key]

        return result_shape


    def evaluate_forward(self, in_shape: 'TensorShape') -> 'TensorShape':
        result_shape = self.copy()

        for dim_key, dim_info in result_shape.dimensions.items():
            dim_size = dim_info.size

            free_symbols = {sym for sym in dim_size.free_symbols}

            end_size = dim_size.subs({
                free_symbol: in_shape[str(free_symbol)]
                for free_symbol
                in free_symbols
                if str(free_symbol) in in_shape
            }).evalf()

            if not end_size.free_symbols and end_size % 1.0 != 0.0:
                raise ValueError(f'Forward evaluation resulted in {end_size = }, should be integer')

            result_shape[dim_key] = end_size

        for dim_key in in_shape.dimensions.keys():
            if dim_key not in result_shape:
                result_shape[dim_key] = in_shape.dimensions[dim_key].size
                continue

        return result_shape

    def evaluate_backward(self, dim_key: DimKeyType, out_shape: 'TensorShape') -> sp.Expr:
        dim_key = _to_key(dim_key)
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
            raise TensorShapeError(f'Backward evaluation of dimension {dim_key} resulted '
                                   f'in {in_size = }, should be integer',
                                   shape=self, out_shape=out_shape)

        return in_size

    def copy(self) -> 'TensorShape':
        return TensorShape(**{
            dim_key: dim_info.size
            for dim_key, dim_info
            in self.dimensions.items()
        })

    def get_definite_features(self) -> int:
        return self.get_definite_size(self.FEATURES_KEY)


def _to_symbol(dim_key: DimKeyType) -> sp.Symbol:
    if isinstance(dim_key, sp.Symbol):
        return dim_key
    return sp.Symbol(dim_key, integer=True)

def _to_key(dim_key: DimKeyType) -> str:
    return str(dim_key)
