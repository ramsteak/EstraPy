from dataclasses import dataclass
from typing import NamedTuple
from matplotlib import colors

import numpy.typing as npt
import numpy as np
import pandas as pd


@dataclass(slots=True, frozen=True)
class CalcToken:
    def calc(self, df:pd.DataFrame) -> pd.Series | float:...

@dataclass(slots=True, frozen=True)
class Func1(CalcToken):
    f:CalcToken
@dataclass(slots=True, frozen=True)
class Func2(CalcToken):
    a:CalcToken
    b:CalcToken

@dataclass(slots=True, frozen=True)
class GetColumn(CalcToken):
    col_x:str
    col_y:str

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(df[self.col_y].to_numpy(), df[self.col_x].to_numpy())

@dataclass(slots=True, frozen=True)
class Value(CalcToken):
    val:float

    def calc(self, df: pd.DataFrame) -> float:
        return self.val

@dataclass(slots=True, frozen=True)
class Product(Func2):

    def calc(self, df: pd.DataFrame) -> pd.Series | float:
        return self.a.calc(df) * self.b.calc(df)

@dataclass(slots=True, frozen=True)
class Sum(Func2):
    
    def calc(self, df: pd.DataFrame) -> pd.Series | float:
        return self.a.calc(df) + self.b.calc(df)

@dataclass(slots=True, frozen=True)
class Derivative(Func1):

    def calc(self, df: pd.DataFrame) -> pd.Series | float:
        _f = self.f.calc(df)
        if isinstance(_f, pd.Series):
            return pd.Series(np.gradient(_f.to_numpy(), _f.index.to_numpy()), _f.index)
        return 0

@dataclass(slots=True, frozen=True)
class ComplexPhase(Func1):
    def calc(self, df:pd.DataFrame) -> pd.Series | float:
        _f = self.f.calc(df)
        if isinstance(_f, pd.Series):
            return pd.Series(np.angle(_f), _f.index)
        return float(np.angle(_f))

@dataclass(slots=True, frozen=True)
class Absolute(Func1):
    def calc(self, df:pd.DataFrame) -> pd.Series | float:
        _f = self.f.calc(df)
        if isinstance(_f, pd.Series):
            return pd.Series(np.abs(_f), _f.index)
        return float(np.abs(_f))

@dataclass(slots=True, frozen=True)
class RealPart(Func1):
    def calc(self, df:pd.DataFrame) -> pd.Series | float:
        _f = self.f.calc(df)
        if isinstance(_f, pd.Series):
            return pd.Series(np.real(_f), _f.index)
        return float(np.real(_f))

@dataclass(slots=True, frozen=True)
class ImagPart(Func1):
    def calc(self, df:pd.DataFrame) -> pd.Series | float:
        _f = self.f.calc(df)
        if isinstance(_f, pd.Series):
            return pd.Series(np.imag(_f), _f.index)
        return float(np.imag(_f))




@dataclass(slots=True, frozen=True)
class PlotType: ...

@dataclass(slots=True, frozen=True)
class XYPlot(PlotType):
    s: CalcToken


class Labels(NamedTuple):
    xlabel: str
    ylabel: str
    title: str
