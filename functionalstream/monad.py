from abc import ABC
from typing import TypeVar, Callable, Iterable

from functionalstream import pipeline


class Monad(ABC):
    T = TypeVar('T')
    R = TypeVar('R')
    Fn_T2T = Callable[[T], T]

    @classmethod
    def unit(cls, x: T) -> R:
        raise NotImplementedError()

    @classmethod
    def bind(cls, function: Callable[[Fn_T2T, T], T], unit_output: R) -> R:
        raise NotImplementedError()

    def __init__(self, functions: Iterable[Fn_T2T]):
        self.functions = functions

    def __call__(self, initializer: T) -> R:
        return pipeline(self.functions, self.__class__.bind)(self.__class__.unit(initializer))
