from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Callable, TypeVar, Iterable, Any, overload, Iterator, Literal

_K = TypeVar('_K')
_R = TypeVar('_R')
ARGS = tuple[Any, ...]
KWRG = dict[str, Any]
ARKW = tuple[ARGS, KWRG]
FUNCARGS = ARGS | KWRG | ARKW | Any
ARGKIND = Literal['a', 'k', 'ak', 's'] | None


def _is_kwargs(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if not all(isinstance(k, str) for k in obj.keys()):  # type: ignore
        return False
    return True


def _is_args(obj: Any) -> bool:
    return isinstance(obj, tuple)


def _is_arkw(obj: Any) -> bool:
    if not isinstance(obj, tuple):
        return False
    if len(obj) != 2:  # type: ignore
        return False
    args, kwargs = obj  # type: ignore
    return _is_args(args) and _is_kwargs(kwargs)


def _normalize_funcargs(funcargs: FUNCARGS, *, kind: ARGKIND = None) -> ARKW:
    """Normalize function arguments into (args, kwargs) format. If kind is specified,
    interpret funcargs accordingly and skip type checks. If kind is None, infer the format from funcargs.
    This could lead to wrong interpretations if the function arguments are ambiguous."""

    # If kind is specified, interpret funcargs accordingly and skip type checks.
    # This can lead to runtime errors if the user is wrong.
    if kind == 'ak':
        return funcargs  # type: ignore [trust the user]
    if kind == 'a':
        return funcargs, {}  # type: ignore [trust the user]
    if kind == 'k':
        return (), funcargs  # type: ignore [trust the user]
    if kind == 's':
        return (funcargs,), {}  # type: ignore [trust the user]

    # Kind is None, infer the format from funcargs.
    if _is_arkw(funcargs):
        return funcargs  # type: ignore
    if _is_args(funcargs):
        return funcargs, {}  # type: ignore
    if _is_kwargs(funcargs):
        return (), funcargs  # type: ignore
    return (funcargs,), {}


def _execute_threaded_list(
    func: Callable[..., _R],
    tasks: Iterable[FUNCARGS],
    *,
    threaded: bool = True,
    argkind: ARGKIND = None,
    pass_key_as: str | None = None,
    max_workers: int | None = None,
) -> list[_R]:
    _funcargs = (_normalize_funcargs(fa, kind=argkind) for fa in tasks)
    if pass_key_as is not None:
        _funcargs = ((ar, kw | {pass_key_as: i}) for i, (ar, kw) in enumerate(_funcargs))

    if threaded:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func, *args, **kwargs) for args, kwargs in _funcargs]
            results = [future.result() for future in futures]
    else:
        results = [func(*args, **kwargs) for args, kwargs in _funcargs]

    return results


def _execute_threaded_dict(
    func: Callable[..., _R],
    tasks: dict[_K, FUNCARGS],
    *,
    threaded: bool = True,
    argkind: ARGKIND = None,
    pass_key_as: str | None = None,
    max_workers: int | None = None,
) -> dict[_K, _R]:
    _funcargs = ((key, _normalize_funcargs(fa, kind=argkind)) for key, fa in tasks.items())
    if pass_key_as is not None:
        _funcargs = ((k, (ar, kw | {pass_key_as: k})) for k, (ar, kw) in _funcargs)

    if threaded:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {key: executor.submit(func, *args, **kwargs) for key, (args, kwargs) in _funcargs}
            results = {key: future.result() for key, future in futures.items()}
    else:
        results = {key: func(*args, **kwargs) for key, (args, kwargs) in _funcargs}

    return results


def _execute_threaded_iter(
    func: Callable[..., _R],
    tasks: Iterable[FUNCARGS],
    *,
    threaded: bool = True,
    argkind: ARGKIND = None,
    pass_key_as: str | None = None,
    max_workers: int | None = None,
) -> Iterator[_R]:
    _funcargs = (_normalize_funcargs(fa, kind=argkind) for fa in tasks)
    if pass_key_as is not None:
        _funcargs = ((ar, kw | {pass_key_as: i}) for i, (ar, kw) in enumerate(_funcargs))

    if threaded:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func, *args, **kwargs) for args, kwargs in _funcargs]
            for future in as_completed(futures):
                yield future.result()
    else:
        for args, kwargs in _funcargs:
            yield func(*args, **kwargs)


def _execute_threaded_dictiter(
    func: Callable[..., _R],
    tasks: dict[_K, FUNCARGS],
    *,
    threaded: bool = True,
    argkind: ARGKIND = None,
    pass_key_as: str | None = None,
    max_workers: int | None = None,
) -> Iterator[tuple[_K, _R]]:
    _funcargs = ((key, _normalize_funcargs(fa, kind=argkind)) for key, fa in tasks.items())
    if pass_key_as is not None:
        _funcargs = ((k, (ar, kw | {pass_key_as: k})) for k, (ar, kw) in _funcargs)

    if threaded:

        def _task_wrapper(key: _K, args: ARGS, kwargs: KWRG) -> tuple[_K, _R]:
            return key, func(*args, **kwargs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_task_wrapper, key, args, kwargs): key for key, (args, kwargs) in _funcargs}
            for future in as_completed(futures):
                yield future.result()
    else:
        for key, (args, kwargs) in _funcargs:
            yield key, func(*args, **kwargs)


@overload
def execute_threaded(
    func: Callable[..., _R],
    tasks: list[FUNCARGS],
    *,
    threaded: bool = True,
    as_complete: Literal[True],
    argkind: ARGKIND = None,
    pass_key_as: str | None = None,
    max_workers: int | None = None,
) -> Iterator[_R]: ...


@overload
def execute_threaded(
    func: Callable[..., _R],
    tasks: list[FUNCARGS],
    *,
    threaded: bool = True,
    as_complete: Literal[False] = False,
    argkind: ARGKIND = None,
    pass_key_as: str | None = None,
    max_workers: int | None = None,
) -> list[_R]: ...


@overload
def execute_threaded(
    func: Callable[..., _R],
    tasks: dict[_K, FUNCARGS],
    *,
    threaded: bool = True,
    as_complete: Literal[True],
    argkind: ARGKIND = None,
    pass_key_as: str | None = None,
    max_workers: int | None = None,
) -> Iterator[_R]: ...


@overload
def execute_threaded(
    func: Callable[..., _R],
    tasks: dict[_K, FUNCARGS],
    *,
    threaded: bool = True,
    as_complete: Literal[False] = False,
    argkind: ARGKIND = None,
    pass_key_as: str | None = None,
    max_workers: int | None = None,
) -> dict[_K, _R]: ...


def execute_threaded(
    func: Callable[..., _R],
    tasks: list[FUNCARGS] | dict[_K, FUNCARGS],
    *,
    threaded: bool = True,
    as_complete: bool = False,
    argkind: ARGKIND = None,
    pass_key_as: str | None = None,
    max_workers: int | None = None,
):
    match tasks, as_complete:
        case t, True if isinstance(t, dict):
            return _execute_threaded_dictiter(
                func, t, threaded=threaded, argkind=argkind, pass_key_as=pass_key_as, max_workers=max_workers
            )
        case t, False if isinstance(t, dict):
            return _execute_threaded_dict(
                func, t, threaded=threaded, argkind=argkind, pass_key_as=pass_key_as, max_workers=max_workers
            )
        case t, True:
            return _execute_threaded_iter(func, t, threaded=threaded, argkind=argkind, max_workers=max_workers)
        case t, False:
            return _execute_threaded_list(func, t, threaded=threaded, argkind=argkind, max_workers=max_workers)
        case _:
            raise TypeError('Invalid tasks type; must be Sequence or dict.')
