from datetime import datetime

from dateutil import parser

now_fn = datetime.utcnow

path_safe_pattern = '%Y-%m-%d_%H-%M-%S_%f'

def get_current_timestamp(path_safe: bool = False) -> str:
    if path_safe:
        return now_fn().strftime(path_safe_pattern)
    return str(now_fn())

def timestamp_to_datetime(timestamp: str):
    if ':' not in timestamp:
        return datetime.strptime(timestamp, path_safe_pattern)
    return parser.parse(timestamp)


def get_current_date() -> str:
    return now_fn().strftime('%Y-%m-%d')
