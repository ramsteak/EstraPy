from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from enum import Enum
from logging import getLogger
from typing import NamedTuple
from dataclasses import dataclass
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import norm
from scipy.interpolate import interp1d

from ._numberunit import NumberUnitRange, NumberUnit, parse_range, actualize_range, parse_nu

from ._context import Context, Column, FourierType, Domain, range_to_index, DataColType
from ._handler import CommandHandler, Token, CommandResult

from ._parser import CommandParser


@dataclass(slots=True, frozen=True)
class Finder:...

@dataclass(slots=True, frozen=True)
class Force(Finder):...

@dataclass(slots=True, frozen=True)
class Median(Finder):
    window: int

@dataclass(slots=True, frozen=True)
class Variance(Finder):
    window: int

@dataclass(slots=True, frozen=True)
class Smooth(Finder):
    window: int

@dataclass(slots=True, frozen=True)
class Polynomial(Finder):
    degree: int

@dataclass(slots=True, frozen=True)
class Even(Finder):...


@dataclass(slots=True, frozen=True)
class Method:
    ...

class Args_Deglitch(NamedTuple):
    bounds: NumberUnitRange
    column: str
    finder: Finder
    method: Method
    pvalue: float


class Deglitch(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Deglitch:
        parser = CommandParser(
            "deglitch", description="Removes and replaces glitched signals."
        )
        parser.add_argument("range", nargs=2)
        parser.add_argument("finder", choices=["force", "median", "variance", "smooth", "polynomial", "even"], default="polynomial")
        parser.add_argument("--window", "-w", type=int)
        parser.add_argument("--degree", "-d", type=int)
        parser.add_argument("--column", "-c", default="I0")
        parser.add_argument("--pvalue", "-p", type=float, default=1e-3)

        # method = parser.add_subparsers(dest="method")
        # remove = method.add_parser("remove")
        # interp = method.add_parser("interpolate")

        args = parser.parse(tokens)

        if args.range is not None:
            _range = parse_range(*args.range)
            if _range.domain == None:
                _range = NumberUnitRange(_range.lower, _range.upper, _range.inter, Domain.REAL)
            if _range.domain != Domain.REAL:
                raise ValueError("Cannot deglitch on non-real domain.")
        else:
            _range = NumberUnitRange(NumberUnit(-np.inf, 0, "eV"), NumberUnit(np.inf, 0, "eV"), None, Domain.REAL)

        match args.finder:
            case "force":
                finder = Force()
            case "median":
                if args.window is None:
                    raise ValueError("Median method requires a window size.")
                if args.degree is not None: raise ValueError("--degree is not valid if method is median")
                finder = Median(args.window)
            case "variance":
                if args.window is None:
                    raise ValueError("Variance method requires a window size.")
                if args.degree is not None: raise ValueError("--degree is not valid if method is variance")
                finder = Variance(args.window)
            case "smooth":
                if args.window is None:
                    raise ValueError("Smooth method requires a window size.")
                if args.degree is not None: raise ValueError("--degree is not valid if method is smooth")
                finder = Smooth(args.window)
            case "polynomial":
                if args.degree is None:
                    raise ValueError("Polynomial method requires a polynomial degree.")
                if args.window is not None: raise ValueError("--window is not valid if method is polynomial")
                finder = Polynomial(args.degree)
            case "even":
                finder = Even()
            case f:
                raise ValueError(f"Invalid finder: {f}")

        return Args_Deglitch(
            _range,
            args.column,
            finder,
            Method(),
            args.pvalue
        )

    @staticmethod
    def execute(args: Args_Deglitch, context: Context) -> CommandResult:
        log = getLogger("deglitch")

        domain = args.bounds.domain or Domain.REAL
        if domain != Domain.REAL:
            raise RuntimeError("Cannot fit preedge to a different domain.")
        
        _axes = [data.get_col_(data.datums[domain].default_axis) for data in context.data] # type: ignore
        range = actualize_range(args.bounds, _axes, "eV")

        for data in context.data:
            idx = range_to_index(data, range)
            X, I = data.get_col_("E"), data.get_col_(args.column)
            x,i = X[idx], I[idx]
            g = np.zeros_like(idx)
            
            match args.finder:
                case Force(): glitch = idx
                case Median(window):
                    pass
                case Variance(window):
                    pass
                case Smooth(window):
                    # Assume I0 is represented by a low frequency trend
                    #    I0 = I0' + eps
                    # Model I0' as a severely smoothed I0
                    s = lowess(i, x, window/len(x), it=0, return_sorted=False)
                    y = i-s

                    # Estimate the std as median of stds
                    std = pd.Series(y,x).rolling(len(X) // 10, center=True).std().median()
                    u = norm.ppf(1-args.pvalue/2) * std
                    g[idx] = np.abs(y) > u
                    pass

                case Polynomial(degree):
                    # Assume I0 is represented by:
                    #    I0 = P_n + eps
                    # Model the I0 trend as a n-polynomial
                    poly = np.poly1d(np.polyfit(x,i, degree))
                    p = poly(x)
                    y = i-p
                    # Estimate the std as median of stds
                    std = pd.Series(y,x).rolling(len(X) // 10, center=True).std().median()
                    u = norm.ppf(1-args.pvalue/2) * std

                    g[idx] = np.abs(y) > u
                    pass
                case Even():
                    # Determine the point difference between consecutive points
                    idxeven = np.indices(x.shape)[0] % 2 == 0
                    xe,ie = x[idxeven], i[idxeven]
                    xo,io = x[~idxeven], i[~idxeven]
                    ieo = interp1d(xe, ie, "cubic", bounds_error=False)(xo)
                    ioe = interp1d(xo, io, "cubic", bounds_error=False)(xe)
                    ii = np.zeros_like(i)
                    ii[idxeven] = ioe
                    ii[~idxeven] = ieo

                    y = i - ii
                    # Estimate the std as median of stds
                    std = pd.Series(y,x).rolling(len(X) // 10, center=True).std().median()
                    u = norm.ppf(1-args.pvalue/2) * std
                    g[idx] = np.abs(y) > u
            pass
            

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError

class Atan(NamedTuple):
    a: float
    b: float
    c: float
    def calc(self, x:npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        s = np.tan(0.4 * np.pi)
        # c is defined to be the shift from b to be 10% - 90%
        return self.a * (np.atan(s * (x-self.b)/self.c) / np.pi + 0.5)
        # return self.a * (np.atan((x-self.b)/self.c) / np.pi + 0.5) # not reparametrized

class ErrorFunction(NamedTuple):
    a: float
    b: float
    c: float
    def calc(self, x:npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        from scipy.special import erf, erfinv
        s = erfinv(0.8)
        return self.a * (1 + erf(s * (x-self.b)/self.c)) * 0.5

class Exponential(NamedTuple):
    a: float
    b: float
    c: float
    def calc(self, x:npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        y = np.zeros_like(x)
        i = x > self.b
        s = np.log(0.1)
        y[i] = self.a * (1 - np.exp(s * (x[i]-self.b)/self.c))
        return y

class Args_MultiEdge(NamedTuple):
    axis: str
    method: Atan|ErrorFunction|Exponential

class MultiEdge(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_MultiEdge:
        parser = CommandParser(
            "multiedge", description="Removes and replaces multiple edges."
        )
        parser.add_argument("--energy", "-E", dest="axis", const="E", action="store_const")
        parser.add_argument("--relenergy", "-e", dest="axis", const="e", action="store_const")
        parser.add_argument("--kvector", "-k", dest="axis", const="k", action="store_const")

        subparsers = parser.add_subparsers(dest="mode")

        arctan = subparsers.add_parser("atan")
        arctan.add_argument("b")
        arctan.add_argument("a")
        arctan.add_argument("c")

        errfun = subparsers.add_parser("erf")
        errfun.add_argument("b")
        errfun.add_argument("a")
        errfun.add_argument("c")

        expfun = subparsers.add_parser("exp")
        expfun.add_argument("b")
        expfun.add_argument("a")
        expfun.add_argument("c")

        args = parser.parse(tokens)

        a = parse_nu(args.a).value
        b = parse_nu(args.b).value
        c = parse_nu(args.c).value

        match args.mode:
            case "atan": mode = Atan(a, b, c)
            case "erf": mode = ErrorFunction(a, b, c)
            case "exp": mode = Exponential(a, b, c)
        
        return Args_MultiEdge(args.axis if args.axis is not None else "E",mode)

    @staticmethod
    def execute(args: Args_MultiEdge, context: Context) -> CommandResult:
        log = getLogger("multiedge")

        for data in context.data:
            X = data.get_col_(args.axis, domain=Domain.REAL)
            Y = data.get_col_("x", domain=Domain.REAL)

            E = args.method.calc(X)
            # TODO: idea --iplot interactive plot:
            # Si apre un plot e un terminale interattivo.
            # L'utente può cambiare i parametri della curva partendo da quelli
            # dati nel file, per ottimizzare il residuo, che viene plottato ad
            # ogni cambio. Alla fine, stampa la linea del comando finale.

            data.datums[Domain.REAL].add_col(E, Column(None, None, DataColType.MULTIEDGE), "mult", False)
            data.datums[Domain.REAL].mod_col("x", Y-E)

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError

