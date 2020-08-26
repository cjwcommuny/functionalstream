from copy import deepcopy
from functools import partial, reduce
from typing import Callable, Optional, Union, List


increment = lambda x: x + 1
always_true = lambda x: True
always_false = lambda x: False
identical = lambda x: x
none = lambda x: None


class Fn(Callable):
    def __init__(self, function: Callable):
        self.fn = function

    def bind(self, *args, **kwargs) -> Callable:
        return partial(self.fn, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __add__(self, other: Union['Pipeline', List[Callable], Callable]) -> 'Pipeline':
        if type(other) == Pipeline:
            return Pipeline([deepcopy(self), deepcopy(other)])
        else:
            return self.__add__(Pipeline(other))


class _Pipe:
    def __init__(self, functions, bind: Callable):
        self.functions = \
            [] if functions is None else \
                [functions] if callable(functions) else \
                    functions
        self.bind = bind

    def __call__(self, value):
        return reduce((lambda v, f: self.bind(f, v)), self.functions, value)



class Pipeline(Fn):
    def __init__(
            self,
            functions: Optional[Union[Callable, List[Callable]]],
            bind: Callable=lambda f, v: f(v)
    ):
        pipe = _Pipe(functions, bind)
        super().__init__(pipe)
        self.functions = pipe.functions


    def append(self, func: Callable) -> 'Pipeline':
        self.functions.append(func)
        return self

    def append_if(self, func: Callable, condition: bool) -> 'Pipeline':
        if condition:
            return self.append(func)
        else:
            return self




