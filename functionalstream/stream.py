import functools
import operator
from collections import Iterable
import itertools
from multiprocessing import Pool
from typing import Optional


def star_fn(f):
    def star_wrapper(tuple_input):
        return f(*tuple_input)
    return star_wrapper



class Stream:
    def __init__(self, iterable: Iterable):
        super().__init__()
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def accumulate(self, func=operator.add, initial=None):
        return Stream(itertools.accumulate(self, func, initial))

    def combinations(self, r):
        return Stream(itertools.combinations(self, r))

    def combinations_with_replacement(self, r):
        return Stream(itertools.combinations_with_replacement(self, r))

    def dropwhile(self, predicate, star=False):
        predicate = star_fn(predicate) if star else predicate
        return Stream(itertools.dropwhile(predicate, self))

    def filterfalse(self, predicate, star=False):
        predicate = star_fn(predicate) if star else predicate
        return Stream(itertools.filterfalse(predicate, self))

    def groupby(self, key=None):
        return Stream(itertools.groupby(self, key))

    def islice(self, *args, **kwargs):
        return Stream(itertools.islice(self, *args, **kwargs))

    def permutations(self, r=None):
        return Stream(itertools.permutations(self, r))

    def repeat(self, times=None):
        return Stream(itertools.repeat(self, times))

    def starmap(self, func, pool: Optional[Pool]=None, chunksize=None):
        if pool is None:
            assert chunksize is None
            return Stream(itertools.starmap(func, self))
        else:
            return Stream(pool.starmap(func, self, chunksize))

    def takewhile(self, predicate, star=False):
        predicate = star_fn(predicate) if star else predicate
        return Stream(itertools.takewhile(predicate, self))

    def tee(self, n=2):
        return Stream(itertools.tee(self, n))

    def enumerate(self, start=0):
        return Stream(enumerate(self, start))

    def filter(self, function, star=False):
        function = star_fn(function) if star else function
        return Stream(filter(function, self))

    def map(self, func, pool: Optional[Pool]=None, chunksize=None):
        if pool is None:
            assert chunksize is None
            return Stream(map(func, self))
        else:
            return Stream(pool.map(func, self, chunksize))

    def reversed(self):
        return Stream(reversed(self.iterable))

    def sorted(self, key=None, reverse=False):
        return Stream(sorted(self, key=key, reverse=reverse))

    def sum(self, start=0):
        return Stream(sum(self, start))

    def reduce(self, function, initializer=None, star=False):
        function = star_fn(function) if star else function
        return Stream(functools.reduce(function, self, initializer))

    def imap(self, pool: Pool, func, chunksize=None, star=False):
        func = star_fn(func) if star else func
        return Stream(pool.imap(func, self, chunksize))

    def imap_unordered(self, pool: Pool, func, chunksize=None, star=False):
        func = star_fn(func) if star else func
        return Stream(pool.imap_unordered(func, self, chunksize))

    def collect(self, function):
        return function(self)

    def to_list(self):
        return list(self)

    def to_tuple(self):
        return tuple(self)

    def to_set(self):
        return set(self)

    def to_frozenset(self):
        return frozenset(self)

    def to_numpy_array(self, *args, **kwargs):
        import numpy as np
        return np.array(self.to_list(), *args, **kwargs)

    def to_tensor(self, *args, **kwargs):
        import torch
        return torch.tensor(self.to_list(), *args, **kwargs)
