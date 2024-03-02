from typing import Any


def all_none_or_all_not_none(*l: Any) -> bool:
    none_count = l.count(None)
    return none_count == 0 or none_count == len(l)


def one_not_none(*l: Any) -> bool:
    none_count = l.count(None)
    return none_count == len(l) - 1


