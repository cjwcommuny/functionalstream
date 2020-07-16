import functools
import operator
from collections import Iterable
import itertools
from multiprocessing import Pool
from typing import Optional, Callable, List


def _star_fn(f):
    def star_wrapper(tuple_input):
        return f(*tuple_input)
    return star_wrapper



class Stream:
    def __init__(self, iterable: Iterable):
        super().__init__()
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def accumulate(self, func: Callable=operator.add, initial=None) -> 'Stream':
        return Stream(itertools.accumulate(self, func, initial))

    def combinations(self, r: int) -> 'Stream':
        return Stream(itertools.combinations(self, r))

    def combinations_with_replacement(self, r: int) -> 'Stream':
        return Stream(itertools.combinations_with_replacement(self, r))

    def dropwhile(self, predicate: Callable, star: bool=False) -> 'Stream':
        predicate = _star_fn(predicate) if star else predicate
        return Stream(itertools.dropwhile(predicate, self))

    def filterfalse(self, predicate: Callable, star: bool=False) -> 'Stream':
        predicate = _star_fn(predicate) if star else predicate
        return Stream(itertools.filterfalse(predicate, self))

    def groupby(self, key=None) -> 'Stream':
        return Stream(itertools.groupby(self, key))

    def islice(self, *args, **kwargs) -> 'Stream':
        return Stream(itertools.islice(self, *args, **kwargs))

    def permutations(self, r: int=None) -> 'Stream':
        return Stream(itertools.permutations(self, r))

    def repeat(self, times: int=None) -> 'Stream':
        return Stream(itertools.repeat(self, times))

    def starmap(self, func: Callable, pool: Optional[Pool]=None, chunksize: int=None) -> 'Stream':
        if pool is None:
            assert chunksize is None
            return Stream(itertools.starmap(func, self))
        else:
            return Stream(pool.starmap(func, self, chunksize))

    def takewhile(self, predicate: Callable, star: bool=False) -> 'Stream':
        predicate = _star_fn(predicate) if star else predicate
        return Stream(itertools.takewhile(predicate, self))

    def tee(self, n: int=2) -> 'Stream':
        return Stream(itertools.tee(self, n))

    def enumerate(self, start: int=0) -> 'Stream':
        return Stream(enumerate(self, start))

    def filter(self, function: Callable, star: bool=False) -> 'Stream':
        function = _star_fn(function) if star else function
        return Stream(filter(function, self))

    def map(self, func: Callable, pool: Optional[Pool]=None, chunksize: int=None) -> 'Stream':
        if pool is None:
            assert chunksize is None
            return Stream(map(func, self))
        else:
            return Stream(pool.map(func, self, chunksize))

    def reversed(self) -> 'Stream':
        return Stream(reversed(self.iterable))

    def sorted(self, key=None, reverse: bool=False) -> 'Stream':
        return Stream(sorted(self, key=key, reverse=reverse))

    def sum(self, start: int=0) -> 'Stream':
        return Stream(sum(self, start))

    def reduce(self, function: Callable, initializer=None, star=False) -> 'Stream':
        function = _star_fn(function) if star else function
        return Stream(functools.reduce(function, self, initializer))

    def imap(self, pool: Pool, func: Callable, chunksize: int=None, star: bool=False) -> 'Stream':
        func = _star_fn(func) if star else func
        return Stream(pool.imap(func, self, chunksize))

    def imap_unordered(self, pool: Pool, func: Callable, chunksize: int=None, star: bool=False) -> 'Stream':
        func = _star_fn(func) if star else func
        return Stream(pool.imap_unordered(func, self, chunksize))

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

    def to_numpy_array(self, *args, **kwargs) -> 'numpy.ndarray':
        import numpy as np
        return np.array(self.to_list(), *args, **kwargs)

    def to_tensor(self, *args, **kwargs) -> 'torch.tensor':
        import torch
        return torch.tensor(self.to_list(), *args, **kwargs)

    def apply(self, function: Callable, star: bool=False) -> None:
        """
        used when function has side effect, e.g. print
        :param function:
        :param star:
        :return:
        """
        if star:
            self.map(function).to_list()
        else:
            self.starmap(function).to_list()
