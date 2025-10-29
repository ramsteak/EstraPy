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


class ColumnKind(Enum):
    AXIS = 'axis'  # e.g. energy, k, r, etc.
    DATA = 'data'  # e.g. I0, mu, chi, etc.
    ERROR = 'error'  # e.g. sI0, smu, etc
    TEMP = 'temp'  # temporary columns, removed after import e.g. magnitude and phase -> complex value


@dataclass(slots=True)
class ColumnDescription:
    name: str
    unit: Unit | None
    type: ColumnKind
    deps: list[str] = field(default_factory=list[str])
    calc: Callable[[pd.DataFrame], pd.Series] | None = None
    labl: str | None = None

@dataclass(slots=True)
class ColumnMetadata:
    versionnumber: int
    physicalname: str
    resolved_deps: list[str]

@dataclass(slots=True)
class Column:
    desc: ColumnDescription
    meta: ColumnMetadata


@dataclass(slots=True)
class DataDomain:
    """A single domain (e.g. reciprocal space, fourier space, etc.) of a data file."""

    columns: dict[str, list[Column]] = field(default_factory=dict[str, list[Column]])
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def add_column(self, name: str, column: ColumnDescription) -> None:
        '''Adds a column to the domain, calculating its data if necessary.'''
        if column.calc is None or not column.deps:
            raise ValueError(f"Column '{name}' has no calculation function or dependencies defined.")
        
        # Resolve dependencies and calculate data
        resolved_deps = self._resolve_columns(column.deps)
        data = column.calc(self.data.loc[:, resolved_deps].rename(columns=dict(zip(resolved_deps, column.deps))))
        self.add_column_data(name, column, data)

    def add_column_data(self, name: str, column: ColumnDescription, data: pd.Series) -> None:
        '''Adds a column to the domain, ignoring the calc attribute of the column
        (i.e. expensive calculations that are already done).'''
        # Ensure that, if the column name already exists, unit matches
        if name in self.columns:
            for ecol in self.columns[name]:
                if ecol.desc.unit != column.unit:
                    raise ValueError(f"Column '{name}' already exists with different unit ({ecol.desc.unit} != {column.unit}).")
        else:
            self.columns[name] = []

        # Resolve dependencies
        resolved_deps = self._resolve_columns(column.deps)
        column_version = self.columns[name][-1].meta.versionnumber +1 if name in self.columns and self.columns[name] else 0
        physicalname = f"{name}.{column_version}"
        colmeta = ColumnMetadata(column_version, physicalname, resolved_deps)

        self.columns[name].append(Column(column, colmeta))
        self.data[physicalname] = data
    
    def _resolve_columns(self, deps: list[str]) -> list[str]:
        '''Resolve dependencies for a column. Returns a list of physical column names.
        If the dependency cannot be resolved, raises a KeyError.'''
        resolved: list[str] = []
        for dep in deps:
            if dep in self.data.columns:
                resolved.append(dep)
            elif dep in self.columns:
                resolved.append(self.columns[dep][-1].meta.physicalname)
            else:
                raise KeyError(f"Cannot resolve dependency '{dep}' for column.")
        return resolved
    
    def get_column(self, name: str) -> Column:
        resolved_name = self._resolve_columns([name])[0]
        return next(col for col in self.columns[name] if col.meta.physicalname == resolved_name)
    
    def get_column_data(self, name: str) -> pd.Series:
        '''Get the latest version of a column by name.'''
        resolved_name = self._resolve_columns([name])[0]
        return self.data[resolved_name].rename(name)
    
    def get_columns_data(self, names: list[str]) -> pd.DataFrame:
        '''Get the latest versions of multiple columns by name.'''
        resolved_names = self._resolve_columns(names)
        return self.data[resolved_names].rename(columns=dict(zip(resolved_names, names)))

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
