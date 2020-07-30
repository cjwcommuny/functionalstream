from functools import partial
from typing import Callable

increment = lambda x: x + 1


class Fn(Callable):
    def __init__(self, function: Callable):
        self.function = function

    def bind(self, *args, **kwargs) -> Callable:
        return partial(self.function, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
