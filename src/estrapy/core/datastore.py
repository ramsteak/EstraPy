import pandas as pd

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .number import Unit

# Structure that hold the file data during the execution of the program, stored in the context variable.


class Domain(Enum):
    RECIPROCAL = 'reciprocal'
    FOURIER = 'fourier'
    PRINCIPALCOMPONENT = 'principalcomponent'
    WAVELET = 'wavelet'


class ColumnType(Enum):
    AXIS = 'axis'  # e.g. energy, k, r, etc.
    DATA = 'data'  # e.g. I0, mu, chi, etc.
    ERROR = 'error'  # e.g. sI0, smu, etc
    TEMP = 'temp'  # temporary columns, removed after import e.g. magnitude and phase -> complex value


@dataclass(slots=True)
class Column:
    name: str
    unit: Unit | None = None
    type: ColumnType = ColumnType.DATA
    calc: Callable[[pd.DataFrame], pd.Series] | None = None


@dataclass(slots=True)
class DataDomain:
    """A single domain (e.g. reciprocal space, fourier space, etc.) of a data file."""

    columns: list[Column] = field(default_factory=list[Column])
    data: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(slots=True)
class FileMetadata:
    path: Path
    name: str
    _dict: dict[str, Any] = field(default_factory=dict[str, Any])

    def __getitem__(self, key: str) -> Any:
        """Get a metadata value by key. Returns None if the key does not exist.
        Also check if the key is an attribute of the class."""
        if hasattr(self, key):
            return getattr(self, key)
        return self._dict[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a metadata value by key. If the key is an attribute of the class,
        set the attribute. Otherwise, set the key in the internal dictionary."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self._dict[key] = value


@dataclass(slots=True)
class DataPage:
    meta: FileMetadata
    domains: dict[Domain, DataDomain] = field(default_factory=dict[Domain, DataDomain])


@dataclass(slots=True)
class DataStore:
    pages: dict[str, DataPage] = field(default_factory=dict[str, DataPage])
