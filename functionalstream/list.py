from typing import Callable

from functionalstream import Stream


class StreamList(Stream):
    def __init__(self, *args):
        super().__init__(list(args))

    def append_if(self, element, condition: bool) -> 'StreamList':
        if condition:
            self.iterable.append(element)
        return self

    def __getattr__(self, name):
        return getattr(self.iterable, name)

