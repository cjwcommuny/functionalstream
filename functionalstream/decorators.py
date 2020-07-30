from functools import wraps
from typing import Callable


def before(before_func: Callable):
    def decorator(main_func: Callable):
        @wraps(main_func)
        def wrapper(*args, **kwargs):
            before_func()
            return main_func(*args, **kwargs)
        return wrapper
    return decorator


def after(after_func: Callable):
    def decorator(main_func: Callable):
        @wraps(main_func)
        def wrapper(*args, **kwargs):
            result = main_func(*args, **kwargs)
            after_func()
            return result
        return wrapper
    return decorator