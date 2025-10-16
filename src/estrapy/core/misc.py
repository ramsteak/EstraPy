from abc import ABC, abstractmethod
from io import TextIOWrapper
from types import EllipsisType
from typing import TypeVar, Generic, Iterable, Collection, Iterator, Any, Literal, Self, Never, Sequence

_K = TypeVar('_K')
_V = TypeVar('_V')
_C = TypeVar('_C', bound=Collection[Any])

class CollectionDict(Collection[_K], Generic[_K, _V, _C], ABC):
    '''Abstract base class for a dictionary mapping keys to collections of values.'''

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
        '''Return a new empty container (list or set).'''

    def add(self, key: _K, value: _V) -> None:
        """Add a value to the collection for the given key."""
        if key not in self._data:
            self._data[key] = self._make_container()
        self._add_value_to_container(self._data[key], value)

    @abstractmethod
    def _add_value_to_container(self, container: _C, value: _V) -> None:
        '''Add a value to the container.'''

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
        '''Remove a value from the container.'''

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
        return f"{self.__class__.__name__}({self._data})"

    def __str__(self) -> str:
        """Return a string representation of the collection dict."""
        return f"{self.__class__.__name__}({self._data})"
    
    def get(self, key: _K) -> _C:
        """Get the collection of values for a given key, or return an empty collection if the key does not exist."""
        if key in self._data:
            return self._data[key]
        return self._make_container()

    @abstractmethod
    def copy(self) -> 'CollectionDict[_K, _V, _C]':
        '''Return a copy of the collection dict.'''

    def clear(self) -> None:
        self._data.clear()


class Bag(CollectionDict[_K, _V, list[_V]]):
    '''A bag is a dictionary that maps keys to lists of values.'''

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
    '''A sack is a dictionary that maps keys to sets of values.'''

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


def peek(f: TextIOWrapper, n: int | Literal["line"] | None = 1) -> str:
    pos = f.tell()
    r = f.readline() if n == "line" else f.read(n)
    f.seek(pos)
    return r

class StaticUtility:
    def __new__(cls, *args: Any, **kwargs: Any) -> Never:
        raise TypeError(f"{cls.__name__} is a static utility class and cannot be instantiated.")

class fmt(StaticUtility):
    """Main namespace for formatting utilities."""
    
    class sup:
        """Namespace for superscript formatting utilities."""
        def __new__(cls, n: int) -> str:
            '''Formats a number in superscript.'''
            return str(n).translate(str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺"))

        @staticmethod
        def th(n: int) -> str:
            '''Formats an integer with its ordinal suffix in superscript (1ˢᵗ, 2ⁿᵈ, 3ʳᵈ, etc.).'''
            if 10 <= n <= 20:
                return f"{n}ᵗʰ"
            suffix = {1: "ˢᵗ", 2: "ⁿᵈ", 3: "ʳᵈ"}.get(n % 10, "ᵗʰ")
            return f"{n}{suffix}"
    
        @staticmethod
        def poly(p: Sequence[float], *, sep:str = " ", var:str = "x", hide_0: bool = True, hide_1: bool = True, order: Literal[-1,+1] = 1, poly_order: Literal[-1,+1] = -1) -> str:
            """Convert a list of polynomial coefficients to a human-readable string."""
            coeff = list(enumerate(p)) if poly_order > 0 else list(enumerate(reversed(p)))
            pieces = [f"{a}{var}{fmt.sup(e)}" for e,a in coeff]
            return sep.join(pieces if order > 0 else reversed(pieces))
    
    class sub:
        """Namespace for subscript formatting utilities."""
        def __new__(cls, n: int) -> str:
            '''Formats a number in subscript.'''
            return str(n).translate(str.maketrans("0123456789-+", "₀₁₂₃₄₅₆₇₈₉₋₊"))

    @staticmethod
    def th(n: int) -> str:
        '''Formats an integer with its ordinal suffix (1st, 2nd, 3rd, etc.).'''
        if 10 <= n <= 20:
            return f"{n}th"
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"
    
    @staticmethod
    def human(n: float, unit:str="", *, format:str|None=None, order: float = 1000, snap: float = 1.00) -> str:
        '''Formats a number with human-readable suffixes (k, M, G, T).'''
        SIprefix = ['', 'k', 'M', 'G', 'T' 'P']
        i,p = 0, ''
        for i, p in enumerate(SIprefix):
            if abs(n) < order**(i+1)*snap:
                break
        if format is None:
            return f"{n/(order**i)}{p}{unit}"
        return f"{n/(order**i):{format}}{p}{unit}"

    @staticmethod
    def poly(p: Sequence[float], *, sep:str = " ", var:str = "x", exp: str = "^", hide_0: bool = True, hide_1: bool = True, order: Literal[-1,+1] = 1, poly_order: Literal[-1,+1] = -1) -> str:
        """Convert a list of polynomial coefficients to a human-readable string."""
        coeff = list(enumerate(p)) if poly_order > 0 else list(enumerate(reversed(p)))
        pieces = [f"{a}{var}{exp}{e}" for e,a in coeff]
        return sep.join(pieces if order > 0 else reversed(pieces))
    