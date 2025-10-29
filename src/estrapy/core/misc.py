import re

from abc import ABC, abstractmethod
from io import TextIOWrapper
from types import EllipsisType
from typing import TypeVar, Generic, Iterable, Collection, Iterator, Any, Literal, Self, NoReturn, Sequence
from collections import deque

from .number import Number, parse_number, Unit
from .edges import edges

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


def peek(f: TextIOWrapper, n: int | Literal['line'] | None = 1) -> str:
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
            sep: str = ' ',
            var: str = 'x',
            order: Literal[-1, +1] = 1,
            poly_order: Literal[-1, +1] = -1,
        ) -> str:
            """Convert a list of polynomial coefficients to a human-readable string."""
            coeff = list(enumerate(p)) if poly_order > 0 else list(enumerate(reversed(p)))
            pieces = [f'{a}{var}{fmt.sup(e)}' for e, a in coeff]
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


def parse_edge(edge: str) -> Number:
    """Parse an edge string and return the energy edge in eV.
    The edge string can be shifted by a number.
    Examples:
        - Pd.K -> K edge of Pd
        - Fe.L3+10eV -> L3 edge of Fe shifted by +10 eV
        - Cu.M5-5eV -> M5 edge of Cu shifted by -5 eV
    """
    try:
        return parse_number(edge)
    except ValueError:
        m = re.match(r'^([A-Za-z]+)(?:\.([A-Za-z0-9+]+))(.*)', edge)
        if m is None:
            raise ValueError(f"Invalid edge format: '{edge}'")

        element, edge, shift = m.groups()
        if shift != '':
            shift = parse_number(shift)
            if shift.unit not in (None, Unit.EV):
                raise ValueError('Edge shift must be in eV.')
            shift = shift.value
        else:
            shift = 0.0

        edgename = f'{element}.{edge}'
        if edgename in edges:
            return Number(None, edges[edgename] + shift, Unit.EV)

        # Use xraydb to get the edge energy if everything else fails
        from xraydb import xray_edge  # type: ignore

        try:
            edge_energy: float = xray_edge(element, edge).energy  # type: ignore
            return Number(edge_energy + shift, Unit.EV)  # type: ignore
        except ValueError:
            raise ValueError(f"Invalid element or edge: '{element}.{edge}'")


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
        """Peek at the next n items without consuming them. Returns a list of the next n items. If there are not enough items, returns the available ones."""
        try:
            while len(self._buffer) < n or n < 1:
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
