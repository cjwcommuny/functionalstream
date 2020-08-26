import functools
import inspect
import itertools
import operator
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from multiprocessing import Pool
from typing import Optional, Callable, TypeVar, Tuple
from functionalstream import functions

def need_star(function: Callable) -> bool:
    return len(inspect.signature(function).parameters) > 1

def _star_fn(f):
    def star_wrapper(tuple_input):
        return f(*tuple_input)
    return star_wrapper

def get_proper_callable(function: Callable, star: Optional[bool]) -> Callable:
    star = need_star(function) if star is None else star
    return _star_fn(function) if star else function


T = TypeVar('T')
Fn_T2T = Callable[[T], T]

class RepeatableIterator(Iterator):
    def __init__(self, iterator):
        super().__init__()
        self.iterator = iterator

    def __iter__(self):
        self.iterator, new_itertor = itertools.tee(self.iterator)
        return new_itertor

    def __next__(self):
        return next(self.iterator)



class Stream(Iterable):
    def __init__(self, iterable: Iterable):
        super().__init__()
        self.iterable = iterable

    def groupby_iterator_type(self):
        return RepeatableIterator

    def __iter__(self):
        self.iterable, new_iterator = itertools.tee(self.iterable)
        return new_iterator

    def accumulate(self, func: Callable=operator.add, initial=None) -> 'Stream':
        return Stream(itertools.accumulate(self, func, initial))

    def combinations(self, r: int) -> 'Stream':
        return Stream(itertools.combinations(self, r))

    def combinations_with_replacement(self, r: int) -> 'Stream':
        return Stream(itertools.combinations_with_replacement(self, r))

    def dropwhile(self, predicate: Callable, star: Optional[bool]=None) -> 'Stream':
        predicate = get_proper_callable(predicate, star)
        return Stream(itertools.dropwhile(predicate, self))

    def filterfalse(self, predicate: Callable, star: Optional[bool]=None) -> 'Stream':
        predicate = get_proper_callable(predicate, star)
        return Stream(itertools.filterfalse(predicate, self))

    def groupby(self, key=None) -> 'Stream':
        return Stream(
            itertools.starmap(
                lambda key, values: (key, self.groupby_iterator_type()(values)),
                itertools.groupby(self, key)
            )
        )

    def slice(self, *args, **kwargs) -> 'Stream':
        return Stream(itertools.islice(self, *args, **kwargs))

    def permutations(self, r: int=None) -> 'Stream':
        return Stream(itertools.permutations(self, r))

    def repeat(self, times: int=None) -> 'Stream':
        return Stream(itertools.repeat(self, times))

    def takewhile(self, predicate: Callable, star: Optional[bool]=None) -> 'Stream':
        predicate = get_proper_callable(predicate, star)
        return Stream(itertools.takewhile(predicate, self))

    def enumerate(self, start: int=0) -> 'Stream':
        return Stream(enumerate(self, start))

    def filter(self, function: Callable, star: Optional[bool]=None) -> 'Stream':
        function = get_proper_callable(function, star)
        return Stream(filter(function, self))

    def map(
            self,
            func: Callable,
            pool: Optional[Pool]=None,
            star: Optional[bool]=None,
            *args,
            **kwargs
    ) -> 'Stream':
        func = get_proper_callable(func, star)
        if pool is None:
            return Stream(map(func, self))
        else:
            return Stream(pool.imap(func, self, *args, **kwargs))

    def reversed(self) -> 'Stream':
        return Stream(reversed(self.iterable))

    def sorted(self, key=None, reverse: bool=False) -> 'Stream':
        return Stream(sorted(self, key=key, reverse=reverse))

    def sum(self, start: int=0) -> 'Stream':
        return Stream(sum(self, start))

    def reduce(self, function: Callable, initializer=None, star: Optional[bool]=None) -> 'Stream':
        function = get_proper_callable(function, star)
        return Stream(functools.reduce(function, self, initializer))

    def imap_unordered(
            self, pool: Pool, func: Callable, star: Optional[bool]=None, *args, **kwargs
    ) -> 'Stream':
        func = get_proper_callable(func, star)
        return Stream(pool.imap_unordered(func, self, *args, **kwargs))

    def collect(self, function: Callable):
        return function(self)

    def to_list(self) -> list:
        return list(self)

    def to_tuple(self) -> tuple:
        return tuple(self)

    def to_set(self) -> set:
        return set(self)

    def to_frozenset(self) -> frozenset:
        return frozenset(self)

    def to_dict(self) -> dict:
        return {key: value for key, value in self}

    def to_ordered_dict(self) -> OrderedDict:
        return OrderedDict(self.to_list())

    def to_numpy_array(self, *args, **kwargs) -> 'numpy.ndarray':
        import numpy as np
        return np.array(self.to_list(), *args, **kwargs)

    def to_tensor(self, *args, **kwargs) -> 'torch.tensor':
        import torch
        return torch.tensor(self.to_list(), *args, **kwargs)

    def cat_to_tensor(self) -> 'torch.Tensor':
        import torch
        return torch.cat(self.to_list())

    def consume(self):
        for _ in self:
            pass

    def foreach(
            self, function: Callable, star: Optional[bool]=None, pool: Optional[Pool]=None, *args, **kwargs
    ) -> 'Stream':
        """
        used when function has side effect, e.g. print
        :param function:
        :param star:
        :param pool
        :return:
        """
        function = get_proper_callable(function, star)
        def f(x):
            return (function(x), x)[1]
        return self.map(f, pool=pool, *args, **kwargs)


    def find_first(self, predicate: Callable, star: Optional[None]=None):
        return next(iter(self.filter(predicate, star)), None)

    def skip(self, start: int) -> 'Stream':
        return self.islice(start=start, stop=None)

    def limit(self, n: int) -> 'Stream':
        return self.islice(stop=n)

    def flatten(self) -> 'Stream':
        return Stream(itertools.chain.from_iterable(self))

    def any(self) -> bool:
        return any(self)

    def all(self) -> bool:
        return all(self)

    def non_empty(self) -> bool:
        return any(True for _ in iter(self))

    def empty(self) -> bool:
        return not self.non_empty()

    def count(self) -> int:
        return sum(1 for _ in self)

    def join_as_str(self, sep: str, str_func: Callable=str, star: Optional[bool]=None):
        star = need_star(str_func) if star is None else star
        if star:
            return sep.join(self.starmap(str_func))
        else:
            return sep.join(self.map(str_func))

    def unpack_tuples(self) -> Tuple[list,...]:
        """
        list[tuple] -> tuple[list]
        """
        return tuple(list(x) for x in zip(*self.to_list()))

    def head(self):
        return next(iter(self))

    def peek_head(self, function: Callable, star: Optional[bool]=None) -> 'Stream':
        function = get_proper_callable(function, star)
        function(self.head())
        return self


class OneOffStream(Stream):
    def __iter__(self):
        return iter(self.iterable)

    def groupby_iterator_type(self):
        return iter
