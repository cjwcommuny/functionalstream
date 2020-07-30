from functools import reduce
from typing import Callable, Iterable, TypeVar

T = TypeVar('T')
Fn_T2T = Callable[[T], T]

def pipeline(
        functions: Iterable[Fn_T2T],
        bind: Callable[[Fn_T2T, T], T]=lambda f, v: f(v)
) -> Fn_T2T:
    """
    Usage:
    >>> from functionalstream.functions import increment
    >>> pipeline([increment, increment, increment])(0)
    3
    """
    def call(value: T) -> T:
        return reduce((lambda v, f: bind(f, v)), functions, value)
    return call
