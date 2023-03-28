import collections
from itertools import repeat


def to_tuple(x, tuple_count: int = 2) -> tuple:
    x = int(x) if isinstance(x, str) else x
    if isinstance(x, collections.abc.Iterable):
        x = [int(i) for i in x]
        return tuple(x)
    return tuple(repeat(x, tuple_count))
