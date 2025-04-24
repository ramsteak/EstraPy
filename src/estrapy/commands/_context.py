import pandas as pd
import numpy as np
import numpy.typing as npt


from pathlib import Path
from typing import Any, NamedTuple
from enum import Enum
from dataclasses import dataclass, field


class Options(NamedTuple):
    interactive: bool
    verbose: bool
    version: tuple[int, ...]
    logcolor: bool
    updateinput: bool
    cache: bool
    other: dict[str, Any]


class Paths(NamedTuple):
    inputfile: Path | None
    currentdir: Path
    workdir: Path
    outputdir: Path
    userdir: Path
    logfile: Path
    configfile: Path


class AxisType(Enum):
    ENERGY = "eV"
    RELENERGY = "+eV"
    KVECTOR = "k"
    DISTANCE = "A"
    QVECTOR = "q"
    INDEX = "i"


class SignalType(Enum):
    FLUORESCENCE = "fl"
    TRANSMITTANCE = "tr"
    INTENSITY = "i"
    OTHER = "o"
    
class FourierType(Enum):
    COMPLEX = "c"
    PHASE = "p"
    ABSOLUTE = "a"
    REALPART = "r"
    IMAGPART = "i"


class DataColType(Enum):
    PREEDGE = "pre"
    POSTEDGE = "post"
    BACKGROUND = "bkg"
    WINDOW = "win"

class Domain(Enum):
    REAL = "real"
    FOURIER = "fourier"
    PCA = "pca"

ColumnType = AxisType | SignalType | DataColType | FourierType

@dataclass(slots=True)
class MetaData:
    signaltype: SignalType | None
    refsigtype: SignalType | None
    name: str
    path: Path
    vars: dict[str, str | int | float]
    run: dict[str, Any] = field(default_factory=dict)
    refE0: float | None = None
    E0: float | None = None

@dataclass(slots=True, frozen=True)
class Column:
    unit: str | None
    axis: ColumnType | None
    

class Datum:
    __slots__ = "df","cols"
    # df: pd.DataFrame
    # cols: dict[str, Column]

    def __init__(self, df:pd.DataFrame|None=None, cols:dict[str,Column]|None=None) -> None:
        match df,cols:
            case None,None:
                self.df = pd.DataFrame()
                self.cols:dict[str, Column] = {}
            case df,cols if df is not None and cols is not None:
                self.df = df
                self.cols = cols
            case _: raise AttributeError("Cannot instanciate if only one of df, cols is None")

    def add_col(self, values:npt.NDArray[np.floating | np.complexfloating] | pd.Series, coltype: Column, colname: str) -> None:
        if colname in self.cols: raise NameError("A column with the given name already exists.")

        self.df.loc[:,colname] = values
        self.cols[colname] = coltype

    def mod_col(self, colname:str, values:npt.NDArray | pd.Series) -> None:
        if colname not in self.cols: raise KeyError(f"No column with the given name: {colname}")
        # Valid column names are "name" (active column) or "name_0", "name_1", ... (old modified column)
        n = max([int(c.split("_", maxsplit=1)[1]) for c in self.cols if c.startswith(f"{colname}_")], default=-1)+1

        old = f"{colname}_{n}"
        self.df.rename(columns={colname:old}, inplace=True)
        self.df.loc[:,colname] = values

        self.cols[old] = self.cols[colname]
    
    def get_cols(self, colname:str=..., *, coltype: ColumnType=...) -> pd.DataFrame: # type: ignore
        if colname is not ...: return self.df.loc[:,[colname]]
        if coltype is not ...: return self.df.loc[:,[c for c,ct in self.cols.items() if ct.axis==coltype]]
        return self.df
    
    # def get_col(self, colname:str) -> pd.Series:
    #     if colname not in self.cols: raise KeyError(f"No column with the given name: {colname}")
    #     return self.df[colname]

    # def get_col_(self, colname:str) -> npt.NDArray[np.floating | np.complexfloating]:
    #     if colname not in self.cols: raise KeyError(f"No column with the given name: {colname}")
    #     return self.get_col(colname).to_numpy()

    # def get_cols_by_type(self, *, unit:str|None=..., axis:ColumnType|None=...) -> pd.DataFrame: # type: ignore
    #     _cols = [*self.cols]
    #     if unit is not ...: _cols = filter(lambda c:self.cols[c].unit == unit, _cols)
    #     if axis is not ...: _cols = filter(lambda c:self.cols[c].axis == axis, _cols)
    #     _cols = [*_cols]
    #     return self.df.loc[:,_cols]
    
    # def get_col_by_type(self, *, unit:str|None=..., axis:ColumnType|None=...) -> pd.Series: # type: ignore
    #     cols = self.get_cols_by_type(unit=unit, axis=axis)
    #     if len(cols.columns) > 1: raise ValueError("More than one columns match")
    #     return cols[cols.columns[0]]
    
    # def get_col_by_type_(self, *, unit:str|None=..., axis:ColumnType|None=...) -> npt.NDArray[np.floating | np.complexfloating]: # type: ignore
    #     return self.get_col_by_type(unit=unit, axis=axis).to_numpy()

class Data:
    def __init__(self, metadata: MetaData) -> None:
        self.datums: dict[Domain, Datum] = {}
        # meta holds the metadata
        self.meta = metadata
    
    def add_col(self, colname:str, values:npt.NDArray | pd.Series, column: Column, domain:Domain) -> None:
        if domain not in self.datums:
            self.datums[domain] = Datum()
        
        self.datums[domain].add_col(values, column, colname)
    
    def _get_col_domain(self, colname:str) -> Domain:
        _ds = [domain for domain, datum in self.datums.items() if colname in datum.cols]
        if len(_ds) == 0: raise KeyError(f"column {colname} was not found")
        elif len(_ds) == 1: return _ds[0]
        else: raise KeyError(f"Multiple matching columns.")

    def mod_col(self, colname:str, values:npt.NDArray | pd.Series) -> None:
        domain = self._get_col_domain(colname)
        self.datums[domain].mod_col(colname, values)

    def get_cols(self, colname:str = ..., *, coltype: ColumnType = ..., domain: Domain = ...) -> pd.DataFrame: # type: ignore
        if domain is not ...:
            return self.datums[domain].get_cols(colname, coltype=coltype)
        for domain,datum in self.datums.items():
            try:
                r = datum.get_cols(colname, coltype=coltype)
            except KeyError: continue
            if len(r.columns): return r
        return pd.DataFrame()

    def get_col(self, colname:str = ..., *, coltype: ColumnType = ..., domain: Domain = ...) -> pd.Series: # type: ignore
        cols = self.get_cols(colname, coltype=coltype, domain=domain)
        if len(cols.columns) > 1: raise ValueError("More than one columns match.")
        return cols[cols.columns[0]]
    def get_col_(self, colname:str = ..., *, coltype: ColumnType = ..., domain: Domain = ...) -> npt.NDArray: # type: ignore
        return self.get_col(colname, coltype=coltype, domain=domain).to_numpy()
    
    def get_xy(self, xname:str, yname:str) -> pd.Series:
        xd = self._get_col_domain(xname)
        yd = self._get_col_domain(yname)
        if xd != yd: raise KeyError("X and Y columns are on different domains.")

        return pd.Series(self.get_col_(yname), self.get_col_(xname), name=yname)


class DataStore:
    def __init__(self) -> None:
        self.data: list[Data] = []

    def add_data(self, data: Data) -> None:
        self.data.append(data)

    def __iter__(self):
        return self.data.__iter__()


class Directives(NamedTuple):
    clear: bool
    noplot: bool


class Context:
    def __init__(
        self,
        paths: Paths,
        config: dict[str, Any],
        options: Options,
        directives: Directives,
    ) -> None:
        self.paths = paths
        self.config = config
        self.options = options
        self.directives = directives

        self.vars: dict[str, Any] = {}
        self.data = DataStore()
