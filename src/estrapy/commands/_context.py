from __future__ import annotations

import pandas as pd
import numpy as np
import numpy.typing as npt

from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
from typing import Any, NamedTuple
from enum import Enum
from dataclasses import dataclass, field
from itertools import chain

from ._numberunit import Domain, NumberUnitRange, Bound, NumberUnit

@dataclass(slots=True, frozen=True)
class CommandResult:
    success: bool | None
    result: Any | None = None
    error: Any | None = None
    warning: Any | None = None


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
    ALPHA = "a"
    CHI = "x"
    MU = "mu"
    PREEDGE = "pre"
    POSTEDGE = "post"
    BACKGROUND = "bkg"
    WINDOW = "win"
    MULTIEDGE = "mult"
    GLITCH = "gli"


class FigureSettings(NamedTuple):
    figurenum: int
    subplot: tuple[int,int]

@dataclass(slots=True)
class AxisRuntime:
    axis: Axes
    xlimits: tuple[NumberUnit | Bound | None, NumberUnit | Bound | None] = None, None
    ylimits: tuple[NumberUnit | Bound | None, NumberUnit | Bound | None] = None, None
    _lines: list[tuple[npt.NDArray, npt.NDArray]] = field(default_factory=list)


@dataclass(slots=True)
class FigureRuntime:
    settings: FigureSettings
    figure: Figure
    axes: dict[tuple[int,int], AxisRuntime]
    shown: bool = False

    def show(self) -> None:
        def on_close(event): self.figure.canvas.stop_event_loop()
        self.figure.canvas.mpl_connect('close_event', on_close)
        self.figure.show()
        self.figure.canvas.start_event_loop(timeout=-1)
        self.shown = True
    
    @classmethod
    def new(cls, settings:FigureSettings) -> FigureRuntime:
        _fig = plt.figure(settings.figurenum)

        match settings.subplot:
            case (1,1):
                _ax = {(1,1):AxisRuntime(_fig.subplots(1,1))}
            case (1,_):
                _axs = _fig.subplots(*settings.subplot)
                _ax = {(1,c):AxisRuntime(_ax) for c,_ax in enumerate(_axs,1)}
            case (_,1):
                _axs = _fig.subplots(*settings.subplot)
                _ax = {(r,1):AxisRuntime(_ax) for r,_ax in enumerate(_axs,1)}
            case (_,_):
                _axss = _fig.subplots(*settings.subplot)
                _ax = {(r,c):AxisRuntime(_ax) for r,_axs in enumerate(_axss,1) for c,_ax in enumerate(_axs,1)}
            case _: raise RuntimeError("Unknown error: #3409234")
            

        figure = FigureRuntime(settings, _fig, _ax, False)
        return figure

class Figures(NamedTuple):
    impl_figsettings: list[FigureSettings]
    expl_figsettings: dict[int, FigureSettings]
    figureruntimes: dict[int, FigureRuntime]

    def get_impl_figurenum(self) -> int:
        # To be used for figures created without --fig flag
        allfignums = chain(self.expl_figsettings, self.figureruntimes)
        usrfignums = filter(lambda f:f <= 1000, allfignums)
        return max(usrfignums, default=1)

    def get_high_figurenum(self) -> int:
        # To be used for figures created by commands. Figures will have fignum > 1000
        allfignums = chain(self.expl_figsettings, self.figureruntimes)
        usrfignums = filter(lambda f:f > 1000, allfignums)
        return max(usrfignums, default=1000)


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

    def __contains__(self, varname:str) -> bool:
        match varname:
            case ".st" if self.signaltype is not None: return True
            case ".rt" if self.refsigtype is not None: return True
            case "E0" if self.E0 is not None: return True
            case "rE0" if self.refE0 is not None: return True
            case var if var in self.vars: return True
        return False


    def get(self, varname:str) -> str | int | float:
        if varname not in self:
            raise RuntimeError(f"Variable {varname} is not defined.")
        match varname:
            case ".st" if self.signaltype is not None:
                return self.signaltype.value
            case ".rt" if self.refsigtype is not None:
                return self.refsigtype.value
            case "E0" if self.E0 is not None:
                return self.E0
            case "rE0" if self.refE0 is not None:
                return self.refE0
            case var if var in self.vars:
                return self.vars[var]
            case var:
                raise RuntimeError(f"Variable {var} is not defined.")

@dataclass(slots=True, frozen=True)
class Column:
    unit: str | None
    sign: bool | None
    axis: ColumnType | None

class Datum:
    __slots__ = "df","cols","default_axis"
    # df: pd.DataFrame
    # cols: dict[str, Column]
    # default_axis: str | None

    def __init__(self, df:pd.DataFrame|None=None, cols:dict[str,Column]|None=None) -> None:
        self.default_axis:str|None = None
        match df,cols:
            case None,None:
                self.df = pd.DataFrame()
                self.cols:dict[str, Column] = {}
            case df,cols if df is not None and cols is not None:
                self.df = df
                self.cols = cols
            case _: raise AttributeError("Cannot instanciate if only one of df, cols is None")

    def set_default_axis(self, colname:str) -> None:
        if colname not in self.cols: raise KeyError(f"Column {colname} does not exist.")
        if not isinstance(self.cols[colname].axis, AxisType):
            raise ValueError(f"Column {colname} is not an axis column.")
        self.default_axis = colname

    def get_default_axis(self) -> pd.Series:
        if self.default_axis is None: raise KeyError("No default axis specified.")
        return self.df[self.default_axis]

    def add_col(self, values:npt.NDArray[np.floating | np.complexfloating] | pd.Series, coltype: Column, colname: str, unique:bool=True) -> str:
        if colname in self.cols:
            if unique:
                raise NameError("A column with the given name already exists.")
            else:
                n = len([c for c in self.cols if c.startswith(colname) and not c.startswith(f"{colname}_")])
                colname = f"{colname}{n}"

        self.df.loc[:,colname] = values
        self.cols[colname] = coltype
        return colname

    def mod_col(self, colname:str, values:npt.NDArray | pd.Series) -> None:
        if colname not in self.cols: raise KeyError(f"No column with the given name: {colname}")
        # Valid column names are "name" (active column) or "name_0", "name_1", ... (old modified column)
        n = max([int(c.split("_", maxsplit=1)[1]) for c in self.cols if c.startswith(f"{colname}_")], default=-1)+1

        old = f"{colname}_{n}"
        self.df.rename(columns={colname:old}, inplace=True)
        self.df.loc[:,colname] = values

        self.cols[old] = self.cols[colname]
    
    def get_cols(self, colname:str|None=..., *, coltype: ColumnType=...) -> pd.DataFrame: # type: ignore
        if colname is None: return self.df.index.to_frame()
        if colname is not ...: return self.df.loc[:,[colname]]
        if coltype is not ...: return self.df.loc[:,[c for c,ct in self.cols.items() if ct.axis==coltype]]
        return self.df

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
        else: raise KeyError("Multiple matching columns.")

    def _get_all_cols(self) -> list[tuple[str, Column, Domain]]:
        return [(colname, coltype, domain)
                for domain,datum in self.datums.items()
                for colname,coltype in datum.cols.items()]

    def mod_col(self, colname:str, values:npt.NDArray | pd.Series) -> None:
        domain = self._get_col_domain(colname)
        self.datums[domain].mod_col(colname, values)

    def get_cols(self, colname:str|None = ..., *, coltype: ColumnType = ..., domain: Domain = ...) -> pd.DataFrame: # type: ignore
        if domain is not ...:
            return self.datums[domain].get_cols(colname, coltype=coltype)
        for domain,datum in self.datums.items():
            try:
                r = datum.get_cols(colname, coltype=coltype)
            except KeyError: continue
            if len(r.columns): return r
        return pd.DataFrame()

    def get_col(self, colname:str|None = ..., *, coltype: ColumnType = ..., domain: Domain = ...) -> pd.Series: # type: ignore
        cols = self.get_cols(colname, coltype=coltype, domain=domain)
        if len(cols.columns) > 1: raise ValueError("More than one columns match.")
        return cols[cols.columns[0]]
    def get_col_(self, colname:str|None = ..., *, coltype: ColumnType = ..., domain: Domain = ...) -> npt.NDArray: # type: ignore
        return self.get_col(colname, coltype=coltype, domain=domain).to_numpy()
    
    def get_xy_(self, xname:str, yname:str) -> tuple[npt.NDArray, npt.NDArray]:
        xd = self._get_col_domain(xname)
        yd = self._get_col_domain(yname)
        if xd != yd: raise KeyError("X and Y columns are on different domains.")

        return self.get_col_(xname, domain = xd), self.get_col_(yname, domain = xd)

    def get_xy(self, xname:str, yname:str) -> pd.Series:
        x,y = self.get_xy_(xname, yname)
        return pd.Series(y, x, name=yname)


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
    vars: dict[str, str]


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
        self.commands: list[tuple[NamedTuple, CommandResult]] = []
        self.data = DataStore()
        self.figures = Figures([], {},{})


def range_to_index(data: Data, range: NumberUnitRange) -> pd.Series:
    if range.domain is None: raise KeyError("Unknown index domain.")
    if range.inter is not None:
        raise ValueError("Index selection cannot be done if interval is not None.")
    if isinstance(range.lower, Bound) or isinstance(range.upper, Bound):
        raise ValueError("Index selection must be done only if the bounds are actualized.")
    
    datum = data.datums[range.domain]
    # Get only the live columns
    cols = [col for col in datum.cols if len(col.split("_")) == 1]
    
    for colname in cols:
        col = datum.cols[colname]
        if range.lower.unit != col.unit: continue
        match col.sign, range.lower.sign:
            case [None, _]:...
            case [False, 0]:...
            case [True, -1|1]:...
            case _: continue
        idx_l = datum.df[colname] >= range.lower.value
        break
    else:
        raise KeyError("No matching column found.")
    
    
    for colname in cols:
        col = datum.cols[colname]
        if range.upper.unit != col.unit: continue
        match col.sign, range.upper.sign:
            case [None, _]:...
            case [False, 0]:...
            case [True, -1|1]:...
            case _: continue
        idx_u = datum.df[colname] <= range.upper.value
        break
    else:
        raise KeyError("No matching column found.")
    
    return idx_l&idx_u

def range_to_index_(data: Data, range: NumberUnitRange) -> npt.NDArray[np.bool_]:
    return range_to_index(data, range).to_numpy()