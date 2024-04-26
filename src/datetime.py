from datetime import datetime


def get_current_timestamp(date_time_seperator: str = '_') -> str:
    return datetime.now().strftime(f'%Y-%m-%d{date_time_seperator}%H.%M.%S')

def get_current_date() -> str:
    return datetime.now().strftime('%Y-%m-%d')
