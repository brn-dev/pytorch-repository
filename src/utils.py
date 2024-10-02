import sys
import traceback
from typing import Any, TypeVar, Optional, Callable

T = TypeVar('T')


def all_none_or_all_not_none(*l: Any) -> bool:
    none_count = l.count(None)
    return none_count == 0 or none_count == len(l)


def one_not_none(*l: Any) -> bool:
    none_count = l.count(None)
    return none_count == len(l) - 1


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
        return class_.__qualname__  # avoid outputs like 'builtins.str'
    return module + '.' + class_.__qualname__


def remove_duplicates_keep_order(l: list[T]) -> list[T]:
    return list(dict.fromkeys(l))


def deep_equals(a, b):
    """
    Recursively checks whether two data structures are equal.
    Supports nested dictionaries, lists, tuples, sets, and basic data types.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        for key in a:
            if not deep_equals(a[key], b[key]):
                return False
        return True
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        for item_a, item_b in zip(a, b):
            if not deep_equals(item_a, item_b):
                return False
        return True
    elif isinstance(a, set) and isinstance(b, set):
        return a == b
    else:
        return a == b


def dict_diff(dict1: dict, dict2: dict):
    result = {}
    all_keys = set(dict1.keys()).union(dict2.keys())
    for key in all_keys:
        val1 = dict1.get(key)
        val2 = dict2.get(key)

        if isinstance(val1, dict) and isinstance(val2, dict):
            sub_diff = dict_diff(val1, val2)
            if sub_diff:
                result[key] = sub_diff
        elif isinstance(val1, list) and isinstance(val2, list):
            sub_diff = list_diff(val1, val2)
            if sub_diff:
                result[key] = sub_diff
        elif val1 != val2:
            result[key] = (val1, val2)
    return result


def list_diff(list1: list, list2: list):
    result = []
    differences_found = False
    max_len = max(len(list1), len(list2))
    for i in range(max_len):
        val1 = list1[i] if i < len(list1) else None
        val2 = list2[i] if i < len(list2) else None

        if isinstance(val1, dict) and isinstance(val2, dict):
            sub_diff = dict_diff(val1, val2)
            result.append(sub_diff if sub_diff else None)
            if sub_diff:
                differences_found = True
        elif isinstance(val1, list) and isinstance(val2, list):
            sub_diff = list_diff(val1, val2)
            result.append(sub_diff if sub_diff else None)
            if sub_diff:
                differences_found = True
        elif val1 != val2:
            result.append((val1, val2))
            differences_found = True
        else:
            result.append(None)
    return result if differences_found else None


