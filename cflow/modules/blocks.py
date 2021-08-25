from typing import Any
from typing import Callable
from functools import wraps


def reuse_fn(f: Callable) -> Callable:
    cache = None

    @wraps(f)
    def cached_fn(*args: Any, _cache: bool = True, **kwargs: Any) -> Callable:
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn
