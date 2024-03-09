
def find_permutation(from_order: list[str], to_order: list[str]) -> list[int]:
    if not set(from_order) == set(to_order):
        raise ValueError(f"from_order ({from_order}) does not contain the same elements as to_order ({to_order}")
    
    return [from_order.index(p) for p in to_order]
