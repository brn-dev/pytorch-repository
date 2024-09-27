import sys
import traceback
from typing import Any, TypeVar, Optional, Callable


def all_none_or_all_not_none(*l: Any) -> bool:
    none_count = l.count(None)
    return none_count == 0 or none_count == len(l)


def one_not_none(*l: Any) -> bool:
    none_count = l.count(None)
    return none_count == len(l) - 1

T = TypeVar('T')
def default_fn(value: Optional[T], _default_fn: Callable[[], T]):
    if value is None:
        return _default_fn()
    return value

def format_current_exception() -> str:
    return ''.join(traceback.format_exception(*sys.exc_info()))

def get_fully_qualified_class_name(obj: Any):
    class_ = obj.__class__
    module = class_.__module__
    if module == 'builtins':
        return class_.__qualname__ # avoid outputs like 'builtins.str'
    return module + '.' + class_.__qualname__

