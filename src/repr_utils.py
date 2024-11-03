import functools
import inspect
import re
from typing import Callable


def func_repr(f: Callable) -> str:
    if f is None:
        return str(None)

    if isinstance(f, functools.partial):
        return func_repr(f.func)

    if f.__name__ == '<lambda>':
        return lambda_repr_from_source(f)

    # repr looks like <built-in method mean of type object at 0x123124512>
    # removing the memory address
    f_repr = re.sub(' at 0x[^>]*', '', repr(f))
    # adding module for full qualification
    f_repr = f_repr[:-1] + f', module={inspect.getmodule(f).__name__}' + f_repr[-1]
    return f_repr


def lambda_repr_from_source(lambda_: Callable):
    try:
        # Get the source code of the lambda function
        source = inspect.getsource(lambda_)
    except OSError:
        return "<Source code not available>"

    # Remove leading and trailing whitespace
    source = source.strip()

    # Find the index where 'lambda' starts
    lambda_index = source.find('lambda')
    if lambda_index == -1:
        return "<Not a lambda function>"

    # Extract the substring starting from 'lambda'
    lambda_str = source[lambda_index:]

    # Define a regex pattern to match the lambda expression
    pattern = r'^lambda\s*(?P<params>[^:]*):\s*(?P<body>.*)(,|\)|$)'
    match = re.match(pattern, lambda_str, re.DOTALL)

    if match:
        params = match.group('params').strip()
        body = match.group('body').strip()

        # Remove enclosing parentheses if they exist
        if body.startswith('(') and body.endswith(')'):
            body = body[1:-1].strip()

        # Return the formatted representation
        if len(params) == 0:
            return f'lambda: {body}'
        else:
            return f'lambda {params}: {body}'
    else:
        return "<Could not parse lambda>"
