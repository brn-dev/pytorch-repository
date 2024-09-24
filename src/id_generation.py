import numpy as np

from src.datetime import get_current_timestamp

ALPHANUMERIC_ALPHABET = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')


def generate_alphanumeric_id(length: int = 6) -> str:
    return ''.join(
        np.random.choice(ALPHANUMERIC_ALPHABET, length)
    )

def generate_timestamp_id(alphanumeric_length: int = 6) -> str:
    id_ = get_current_timestamp(path_safe=True)

    if alphanumeric_length > 0:
        id_ += '~' + generate_alphanumeric_id(alphanumeric_length)

    return id_
