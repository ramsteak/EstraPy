import re

from abc import ABC, abstractmethod
from io import TextIOWrapper
from types import EllipsisType
from typing import TypeVar, Generic, Iterable, Collection, Iterator, Any, Literal, Self, NoReturn, Sequence, Protocol, overload
from collections import deque
from enum import Enum

from .number import Number, parse_number, Unit

_K = TypeVar('_K')
_V = TypeVar('_V')
_C = TypeVar('_C', bound=Collection[Any])


class CollectionDict(Collection[_K], Generic[_K, _V, _C], ABC):
    """Abstract base class for a dictionary mapping keys to collections of values."""

    def __init__(self, _dict: dict[_K, _C] | None = None) -> None:
        """Initialize the collection dict."""
        self._data: dict[_K, _C] = {}
        if _dict:
            for k, vs in _dict.items():
                for v in vs:
                    self.add(k, v)

    @classmethod
    def from_iter(cls: type[Self], iterable: Iterable[tuple[_K, _V]]) -> Self:
        """Create a CollectionDict from an iterable of key-value pairs."""
        obj = cls()
        for k, v in iterable:
            obj.add(k, v)
        return obj

    def __getitem__(self, key: _K) -> _C:
        """Get the collection of values for a given key. If the key does not exist, raise KeyError."""
        return self._data[key]

    @abstractmethod
    def _make_container(self) -> _C:
        """Return a new empty container (list or set)."""

    def add(self, key: _K, value: _V) -> None:
        """Add a value to the collection for the given key."""
        if key not in self._data:
            self._data[key] = self._make_container()
        self._add_value_to_container(self._data[key], value)
    
    def adds(self, key: _K, values: Iterable[_V]) -> None:
        """Add multiple values to the collection for the given key."""
        if key not in self._data:
            self._data[key] = self._make_container()
        for value in values:
            self._add_value_to_container(self._data[key], value)
    
    def add_empty(self, key: _K) -> None:
        """Ensure that the key exists in the collection dict with an empty container."""
        if key not in self._data:
            self._data[key] = self._make_container()
    
    def remove_empty(self, key: _K) -> None:
        """Remove the key from the collection dict if its container is empty."""
        if key in self._data and not self._data[key]:
            del self._data[key]
    
    def cull_empty(self) -> None:
        """Remove all keys from the collection dict whose containers are empty."""
        keys_to_remove = [k for k, v in self._data.items() if not v]
        for k in keys_to_remove:
            del self._data[k]

    @abstractmethod
    def _add_value_to_container(self, container: _C, value: _V) -> None:
        """Add a value to the container."""

    def __delitem__(self, key: _K) -> None:
        """Delete the collection of values for a given key."""
        del self._data[key]

    def remove(self, key: _K, value: _V) -> None:
        """Remove a value from the collection for the given key. If the value is the last in the collection, remove the key."""
        if key in self:
            try:
                self._remove_value_from_container(self._data[key], value)
                if not self._data[key]:
                    del self._data[key]
            except ValueError:
                pass

    @abstractmethod
    def _remove_value_from_container(self, container: _C, value: _V) -> None:
        """Remove a value from the container."""

    def __iter__(self) -> Iterator[_K]:
        """Iterate over the keys in the collection dict."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of keys in the collection dict."""
        return len(self._data)

    def count_keys(self) -> int:
        """Return the number of keys in the collection dict."""
        return len(self._data)

    def count(self, key: _K | EllipsisType = ...) -> int:
        """Return the number of values for a given key, or the total number of values if key is not provided."""
        if key is ...:
            return sum(len(vs) for vs in self._data.values())
        if key in self._data:
            return len(self._data[key])
        return 0

    def __contains__(self, key: object) -> bool:
        """Check if a key is in the collection dict."""
        return key in self._data

    def key_contained(self, key: _K) -> bool:
        """Check if a key is in the collection dict."""
        return key in self._data

    def item_contained(self, key: _K, value: _V) -> bool:
        """Check if a value is in the collection for a given key."""
        if key in self._data:
            return value in self._data[key]
        return False

    def groups(self) -> Iterable[tuple[_K, _C]]:
        """Iterate over the key-collection pairs in the collection dict."""
        return self._data.items()

    def items(self) -> Iterable[tuple[_K, _V]]:
        """Iterate over the key-value pairs in the collection dict."""
        yield from ((k, v) for k, vs in self._data.items() for v in vs)

    def __repr__(self) -> str:
        """Return a string representation of the collection dict."""
        return f'{self.__class__.__name__}({self._data})'

    def __str__(self) -> str:
        """Return a string representation of the collection dict."""
        return f'{self.__class__.__name__}({self._data})'

    def get(self, key: _K) -> _C:
        """Get the collection of values for a given key, or return an empty collection if the key does not exist."""
        if key in self._data:
            return self._data[key]
        return self._make_container()
    
    def set(self, key: _K, values: _C) -> None:
        """Set the collection of values for a given key. Overwrites any existing collection,
        does not check for the type of the collection."""
        self._data[key] = values

    @abstractmethod
    def copy(self) -> 'CollectionDict[_K, _V, _C]':
        """Return a copy of the collection dict."""

    def clear(self) -> None:
        self._data.clear()


class Bag(CollectionDict[_K, _V, list[_V]]):
    """A bag is a dictionary that maps keys to lists of values."""

    def _make_container(self) -> list[_V]:
        return []

    def _add_value_to_container(self, container: list[_V], value: _V) -> None:
        container.append(value)

    def _remove_value_from_container(self, container: list[_V], value: _V) -> None:
        container.remove(value)

    def copy(self) -> 'Bag[_K, _V]':
        new_bag = Bag[_K, _V]()
        for k, vs in self._data.items():
            new_bag._data[k] = vs.copy()
        return new_bag


class Sack(CollectionDict[_K, _V, set[_V]]):
    """A sack is a dictionary that maps keys to sets of values."""

    def _make_container(self) -> set[_V]:
        return set()

    def _add_value_to_container(self, container: set[_V], value: _V) -> None:
        container.add(value)

    def _remove_value_from_container(self, container: set[_V], value: _V) -> None:
        container.remove(value)

    def copy(self) -> 'Sack[_K, _V]':
        new_sack = Sack[_K, _V]()
        for k, vs in self._data.items():
            new_sack._data[k] = vs.copy()
        return new_sack


def peek_text(f: TextIOWrapper, n: int | Literal['line'] | None = 1) -> str:
    pos = f.tell()
    r = f.readline() if n == 'line' else f.read(n)
    f.seek(pos)
    return r


class StaticUtility:
    def __new__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError(f'{cls.__name__} is a static utility class and cannot be instantiated.')


class fmt(StaticUtility):
    """Main namespace for formatting utilities."""

    class sup:
        """Namespace for superscript formatting utilities."""

        def __new__(cls, n: int) -> str:
            """Formats a number in superscript."""
            return str(n).translate(str.maketrans('0123456789-+', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺'))

        @staticmethod
        def th(n: int) -> str:
            """Formats an integer with its ordinal suffix in superscript (1ˢᵗ, 2ⁿᵈ, 3ʳᵈ, etc.)."""
            if 10 <= n <= 20:
                return f'{n}ᵗʰ'
            suffix = {1: 'ˢᵗ', 2: 'ⁿᵈ', 3: 'ʳᵈ'}.get(n % 10, 'ᵗʰ')
            return f'{n}{suffix}'

        @staticmethod
        def poly(
            p: Sequence[float],
            *,
            floatfmt: str='',
            sep: str = ' ',
            var: str = 'x',
            order: Literal[-1, +1] = 1,
            poly_order: Literal[-1, +1] = -1,
        ) -> str:
            """Convert a list of polynomial coefficients to a human-readable string."""
            coeff = list(enumerate(p)) if poly_order > 0 else list(enumerate(reversed(p)))
            pieces = [f'{format(a, floatfmt)}{var}{fmt.sup(e)}' for e, a in coeff]
            return sep.join(pieces if order > 0 else reversed(pieces))

    class sub:
        """Namespace for subscript formatting utilities."""

        def __new__(cls, n: int) -> str:
            """Formats a number in subscript."""
            return str(n).translate(str.maketrans('0123456789-+', '₀₁₂₃₄₅₆₇₈₉₋₊'))

    @staticmethod
    def th(n: int) -> str:
        """Formats an integer with its ordinal suffix (1st, 2nd, 3rd, etc.)."""
        if 10 <= n <= 20:
            return f'{n}th'
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f'{n}{suffix}'

    @staticmethod
    def human(n: float, unit: str = '', *, format: str | None = None, order: float = 1000, snap: float = 1.00) -> str:
        """Formats a number with human-readable suffixes (k, M, G, T)."""
        SIprefix = ['', 'k', 'M', 'G', 'T' 'P']
        i, p = 0, ''
        for i, p in enumerate(SIprefix):
            if abs(n) < order ** (i + 1) * snap:
                break
        if format is None:
            return f'{n/(order**i)}{p}{unit}'
        return f'{n/(order**i):{format}}{p}{unit}'

    @staticmethod
    def poly(
        p: Sequence[float],
        *,
        sep: str = ' ',
        var: str = 'x',
        exp: str = '^',
        after_exp: str = '',  # string to add after the exponent, useful for <sup> or <sub>
        order: Literal[-1, +1] = 1,
        poly_order: Literal[-1, +1] = -1,
    ) -> str:
        """Convert a list of polynomial coefficients to a human-readable string."""
        coeff = list(enumerate(p)) if poly_order > 0 else list(enumerate(reversed(p)))
        pieces: list[str] = []
        pieces = [f'{a}{var}{exp}{e}{after_exp}' for e, a in coeff]
        return sep.join(pieces if order > 0 else reversed(pieces))

    @staticmethod
    def prettytable(rows: Iterable[Iterable[Any]], sep: str = '│', header: bool = False) -> str:
        """Formats a table from an iterable of rows."""
        # ┌─┬─┐    0  1  2  3  4
        # │ │ │    5     7     9
        # ├─┼─┤   10 11 12    14
        # │ │ │   15    17    19
        # └─┴─┘   20 21 22    24
        # Convert all items to string
        str_rows = [[f' {item!s} ' for item in row] for row in rows]
        # Calculate the maximum width of each column
        col_widths = [max(len(row[i]) for row in str_rows) for i in range(len(str_rows[0]))]
        # Create the formatted table
        lines: list[str] = []

        if header:
            lines.append('┌' + ('┬'.join('─' * col_widths[j] for j in range(len(str_rows[0])))) + '┐')

        for i, row in enumerate(str_rows):
            lines.append('│' + (sep.join(item.ljust(col_widths[j]) for j, item in enumerate(row))) + '│')
            if header and i == 0:
                lines.append('├' + ('┼'.join('─' * col_widths[j] for j in range(len(row)))) + '┤')

        lines.append('└' + ('┴'.join('─' * col_widths[j] for j in range(len(str_rows[0])))) + '┘')

        return '\n'.join(lines)

    @staticmethod
    def mdtable(rows: Iterable[Iterable[Any]]) -> str:
        """Formats a markdown table from an iterable of rows."""
        str_rows = [[f' {item!s} ' for item in row] for row in rows]
        col_widths = [max(len(row[i]) for row in str_rows) for i in range(len(str_rows[0]))]
        lines: list[str] = []

        for i, row in enumerate(str_rows):
            lines.append('|' + '|'.join(item.ljust(col_widths[j]) for j, item in enumerate(row)) + '|')
            if i == 0:
                lines.append('|' + '|'.join('-' * col_widths[j] for j in range(len(row))) + '|')
        return '\n'.join(lines)



_T = TypeVar('_T')


class peekable(Iterator[_T], Generic[_T]):
    def __init__(self, iterator: Iterator[_T] | Iterable[_T]) -> None:
        self._iterator = iter(iterator)
        self._buffer: deque[_T] = deque()
        self._iterator_exhausted = False

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> _T:
        if self._buffer:
            return self._buffer.popleft()
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator_exhausted = True
            raise

    def peek(self) -> _T:
        """Peek at the next item without consuming it. Returns the next item. If there are no more items, raises StopIteration."""
        if self._buffer:
            return self._buffer[0]
        try:
            item = next(self._iterator)
            self._buffer.append(item)
        except StopIteration:
            self._iterator_exhausted = True
            raise
        return item

    def peek_n(self, n: int) -> list[_T]:
        """Peek at the next n items without consuming them. Returns a list of the next n items.
        If there are not enough items, returns the available ones.
        If n is negative, consume the entire iterator and return all remaining items."""
        try:
            while len(self._buffer) < n or n < 1: # If n < 1, consume the entire iterator
                item = next(self._iterator)
                self._buffer.append(item)
        except StopIteration:
            self._iterator_exhausted = True
        return list(self._buffer)[:n]

    def next(self) -> _T:
        """Get the next item, consuming it. If there are no more items, raises StopIteration."""
        return self.__next__()

    def next_n(self, n: int) -> list[_T]:
        """Get the next n items, consuming them. Returns a list of the next n items. If there are not enough items, returns the available ones."""
        items: list[_T] = []
        try:
            for _ in range(n):
                items.append(self.__next__())
        except StopIteration:
            self._iterator_exhausted = True
        return items

    def pushback(self, item: _T) -> None:
        """Push an item back to the front of the iterator."""
        self._buffer.appendleft(item)

    def pushback_n(self, items: Iterable[_T]) -> None:
        """Push multiple items back to the front of the iterator."""
        for item in reversed(list(items)):
            self._buffer.appendleft(item)

    def __bool__(self) -> bool:
        """Return True if there are more items to iterate over, False otherwise."""
        try:
            self.peek()
            return True
        except StopIteration:
            return False

    def __str__(self) -> str:
        if self._buffer:
            return f'[{str(list(self._buffer))[1:-1]}{', <...>' if not self._iterator_exhausted else ''}]'
        if self._iterator_exhausted:
            return '[<>]'
        return '[<...>]'

    def __repr__(self) -> str:
        if self._buffer:
            return f'peekable([{repr(list(self._buffer))[1:-1]}{', <...>' if not self._iterator_exhausted else ''}])'
        if self._iterator_exhausted:
            return 'peekable([<>])'
        return 'peekable([<...>])'

    def close(self) -> None:
        """Close the underlying iterator if it has a close method."""
        if hasattr(self._iterator, 'close'):
            self._iterator.close()  # type: ignore


from typing import Iterable, Literal, overload, Union

@overload
def fuzzy_match(
    string: str,
    options: Iterable[Union[str, tuple[str, ...]]],
    *,
    strict: Literal[True],
    min_length: int = 1,
) -> str: ...

@overload
def fuzzy_match(
    string: str,
    options: Iterable[Union[str, tuple[str, ...]]],
    *,
    strict: Literal[False] = False,
    min_length: int = 1,
) -> str | None: ...

def fuzzy_match(
    string: str,
    options: Iterable[Union[str, tuple[str, ...]]],
    *,
    strict: bool = False,
    min_length: int = 1,
) -> str | None:
    """Perform a fuzzy match of a string against a list of options.
    
    Returns the best matching option (or the first element if option is a tuple).
    The matching is case-insensitive, ignores underscores, hyphens and spaces.
    
    In order to match, a string must not contain any character that is not in the option,
    and the characters must appear in the same order.
    
    Args:
        string: The string to match against options
        options: Iterable of strings or tuples of strings (aliases for the same option)
        strict: If True, raises ValueError when no match is found
        min_length: Minimum length required for the input string (default: 1)
    
    Returns:
        The matched option string (first element if tuple), or None if no match
    
    Raises:
        ValueError: If strict=True and no match is found
        ValueError: If min_length < 1
        ValueError: If string length < min_length
        ValueError: If options is empty
        ValueError: If any option or tuple element is empty
    
    Examples:
        >>> options = ['fouriertransform', 'background', 'postedge']
        >>> fuzzy_match('fou-tran', options)
        'fouriertransform'
        
        >>> # Using aliases
        >>> options = [('color', 'colour'), 'background']
        >>> fuzzy_match('colour', options)
        'color'
        
        >>> # Minimum length requirement
        >>> fuzzy_match('f', options, min_length=3)
        Traceback (most recent call last):
        ...
        ValueError: Input string "f" is too short (minimum length: 3)
    """
    # Validation
    if min_length < 1:
        raise ValueError(f"min_length must be at least 1, got {min_length}")
    
    if len(string) < min_length:
        raise ValueError(
            f'Input string "{string}" is too short (minimum length: {min_length})'
        )
    
    # Normalize options: convert to list and handle tuples
    options_list = list(options)
    if not options_list:
        raise ValueError("Options cannot be empty")
    
    # Build mapping: normalized_option -> (canonical_return_value, all_aliases)
    # canonical_return_value is what we return (first element if tuple)
    option_map: dict[str, tuple[str, list[str]]] = {}
    
    for option in options_list:
        if isinstance(option, tuple):
            if not option:
                raise ValueError("Option tuple cannot be empty")
            if any(not alias or not isinstance(alias, str) for alias in option): # pyright: ignore[reportUnnecessaryIsInstance]
                raise ValueError("All tuple elements must be non-empty strings")
            
            canonical = option[0]
            aliases = list(option)
        else:
            if not option or not isinstance(option, str): # pyright: ignore[reportUnnecessaryIsInstance]
                raise ValueError("Options must be non-empty strings or tuples of strings")
            canonical = option
            aliases = [option]
        
        # Map all normalized aliases to the canonical form
        for alias in aliases:
            normalized = _normalize_string(alias)
            if normalized in option_map:
                raise ValueError(
                    f'Duplicate option detected: "{alias}" conflicts with existing option'
                )
            option_map[normalized] = (canonical, aliases)
    
    # Normalize target string
    target = _normalize_string(string)
    
    # Check for exact match first
    if target in option_map:
        return option_map[target][0]
    
    # Fuzzy matching
    scores: dict[str, float] = {}
    
    for normalized_option, (canonical, _) in option_map.items():
        idxs: list[int] = []
        for char in target:
            pos = normalized_option.find(char, idxs[-1] + 1 if idxs else 0)
            if pos == -1:
                idxs.clear()
                break
            idxs.append(pos)
        
        if not idxs:
            continue
        
        # Scoring: prefer matches that:
        # 1. Start earlier (idxs[0])
        # 2. Have characters closer together (sum of gaps)
        # 3. Are shorter overall (1 - 1/len penalty)
        # Lower score is better
        scores[normalized_option] = sum([
            idxs[0],  # Penalty for not starting at beginning
            sum(a - b - 1 for a, b in zip(idxs[1:], idxs[:-1])),  # Sum of gaps
            1 - 1 / len(normalized_option)  # Length penalty (shorter is better)
        ])
    
    if not scores:
        if strict:
            alias_strs: list[str] = []
            for canonical, aliases in option_map.values():
                if len(aliases) > 1:
                    alias_strs.append(f"{aliases[0]} (aliases: {', '.join(aliases[1:])})")
                else:
                    alias_strs.append(aliases[0])
            raise ValueError(
                f'No match found for "{string}" in options: {alias_strs}'
            )
        return None
    
    best_match = min(scores, key=scores.__getitem__)
    return option_map[best_match][0]


def _normalize_string(s: str) -> str:
    """Normalize a string for matching: lowercase and remove separators."""
    return s.casefold().replace('_', '').replace('-', '').replace(' ', '')

_E = TypeVar('_E', bound=Enum)

@overload
def fuzzy_match_enum(string: str, enum: type[_E], *, strict: Literal[True]) -> _E: ...
@overload
def fuzzy_match_enum(string: str, enum: type[_E], *, strict: Literal[False] = False) -> _E | None: ...

def fuzzy_match_enum(string: str, enum: type[_E], *, strict: bool = False) -> _E | None:
    match = fuzzy_match(string, [e.name for e in enum], strict=strict)
    if match is None:
        return None
    return enum[match]


def guess_type(s: str) -> str | Number | float | int:
    # Try to guess if the string is an integer, a number with unit, or just a string
    try:
        return int(s)
    except ValueError:
        pass
    try:
        n = parse_number(s)
        if n.unit is None:
            return n.value
        return n
    except ValueError:
        pass
    return s

_K_con = TypeVar('_K_con', contravariant=True)
_V_cov = TypeVar('_V_cov', covariant=True)
class SupportsGetItem(Protocol[_K_con, _V_cov]):
    def __getitem__(self, key: _K_con) -> _V_cov: ...

RE_VARNAME = re.compile(r"\{([^{:}]*)(?::([^{:}]*))?\}")
def template_replace(template: str, vars: SupportsGetItem[str, str]) -> str:
    """Replace variables in a template string with values from a mapping.

    Variables are enclosed in curly braces {}. Optionally, a format specifier can be provided after a colon :.

    Example:
        template = "Hello, {name}! You have {count:d} new messages."
        vars = {'name': 'Alice', 'count': '5'}
        result = template_replace(template, vars)
        # result == "Hello, Alice! You have 5 new messages."
    """
    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        format_spec = match.group(2) or ''
        value = vars[var_name]
        if format_spec:
            return format(value, format_spec)
        return str(value)

    return RE_VARNAME.sub(replacer, template)


from ..core.datastore import Domain
def infer_axis_domain(
        *,
        axis: str | None = None,
        range: tuple[Number, Number] | None = None,
        numbers: list[Number] | None = None,
        domain: Domain | None = None,
        
        ) -> tuple[str, Domain]:
        """Infer the axis and domain from the range, domain and other numbers. Returns a tuple of (axis, domain)."""
        domains: dict[str, Domain] = {
            'E': Domain.RECIPROCAL, 'e': Domain.RECIPROCAL, 'k': Domain.RECIPROCAL, 'q': Domain.RECIPROCAL,
            'r': Domain.FOURIER}
        units: dict[str, Unit] = {'E': Unit.EV, 'e': Unit.EV, 'k': Unit.K, 'q': Unit.K, 'r': Unit.A}
        sign: dict[str, bool] = {'E': False, 'e': True} # All others are None (no sign restriction). E requires no sign, e requires sign.
        # If axis is provided, it should match the domain and the unit of range and numbers
        # If the value of the ranges are -np.inf or np.inf the unit could be None

        if axis is not None:
            expected_domain = domains.get(axis)
            expected_unit = units.get(axis)
            expected_sign = sign.get(axis)

            if expected_domain is None:
                raise ValueError(f"Cannot infer domain from axis '{axis}'.")
            if domain is not None and domain != expected_domain:
                raise ValueError(f"Axis '{axis}' is incompatible with domain '{domain.name}'.")
            if range is not None:
                range_low, range_high = range
                if range_low.unit is not None and range_low.unit != expected_unit:
                    raise ValueError(f"Axis '{axis}' is incompatible with lower range unit '{range_low.unit.name}'.")
                if range_low.sign is None and expected_sign is True:
                    raise ValueError(f"Axis '{axis}' requires sign for lower range, but none was provided.")
                
                if range_high.unit is not None and range_high.unit != expected_unit:
                    raise ValueError(f"Axis '{axis}' is incompatible with upper range unit '{range_high.unit.name}'.")
                if range_high.sign is None and expected_sign is True:
                    raise ValueError(f"Axis '{axis}' requires sign for upper range, but none was provided.")
                
            if numbers is not None:
                for number in numbers:
                    if number.unit is not None and number.unit != expected_unit:
                        raise ValueError(f"Axis '{axis}' is incompatible with number unit '{number.unit.name}'.")
                    # Do not check sign for numbers, as they may represent intervals or other constructs.
            return axis, expected_domain
        
        # If axis is not provided, try to infer it from the domain and the unit of range and numbers.
        # Inferred axis is a list of possible options. We join them at the end to get the best match,
        # if there are multiple options left, we choose based on the order.
        inferred_axis: list[set[str]] = []

        if domain is not None:
            possible_axes = {ax for ax, dom in domains.items() if dom == domain}
            if not possible_axes:
                raise ValueError(f"Cannot infer axis from domain '{domain.name}'.")
            inferred_axis.append(possible_axes)
        
        if range is not None:
            range_low, range_high = range
            possible_axes = set(domains.keys())
            if range_low.unit is not None:
                possible_axes &= {ax for ax, unit in units.items() if unit == range_low.unit}
            if range_low.sign is not None:
                possible_axes &= {ax for ax in sign.items()}
            
            if range_high.unit is not None:
                possible_axes &= {ax for ax, unit in units.items() if unit == range_high.unit}
            if range_high.sign is not None:
                possible_axes &= {ax for ax in sign.items()}
            
            if not possible_axes:
                raise ValueError(f"Cannot infer axis from range '{range_low}' to '{range_high}'.")
            inferred_axis.append(possible_axes)
        
        if numbers is not None:
            possible_axes = set(domains.keys())
            for number in numbers:
                if number.unit is not None:
                    possible_axes &= {ax for ax, unit in units.items() if unit == number.unit}
            if not possible_axes:
                raise ValueError(f"Cannot infer axis from numbers with units {[n.unit for n in numbers]}.")
            inferred_axis.append(possible_axes)
        
        if not inferred_axis:
            raise ValueError("Cannot infer axis without any information.")
        
        common_axes = set[str].intersection(*inferred_axis)
        if not common_axes:
            raise ValueError("Inconsistent information provided, cannot infer axis.")
        
        # Choose the best match based on the order of domains
        for ax in domains.keys():
            if ax in common_axes:
                return ax, domains[ax]
        
        raise ValueError("Cannot infer axis.")