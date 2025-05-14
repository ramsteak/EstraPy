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
class Finder_Force(Finder):...

@dataclass(slots=True, frozen=True)
class Finder_Variance(Finder):
    width: int
    pvalue: float

@dataclass(slots=True, frozen=True)
class Finder_Smooth(Finder):
    width: int
    fraction: float
    pvalue: float

@dataclass(slots=True, frozen=True)
class Finder_Polynomial(Finder):
    width: int
    degree: int
    pvalue: float

@dataclass(slots=True, frozen=True)
class Finder_Even(Finder):
    pvalue: float


@dataclass(slots=True, frozen=True)
class Method:
    noise:bool

@dataclass(slots=True, frozen=True)
class Method_Remove(Method):
    ...

@dataclass(slots=True, frozen=True)
class Method_Base(Method):
    ...

@dataclass(slots=True, frozen=True)
class Method_Smooth(Method):
    fraction: float



class Args_Deglitch(NamedTuple):
    bounds: NumberUnitRange
    column: str
    finder: Finder
    method: Method


class Deglitch(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Deglitch:
        parser = CommandParser(
            "deglitch", description="Removes and replaces glitched signals."
        )
        parser.add_argument("range", nargs=2)
        parser.add_argument("--column", "-c", default="I0")
        # parser.add_argument("finder", choices=["force", "variance", "smooth", "polynomial", "even"], default="polynomial")
        # parser.add_argument("--window", "-w", type=int)
        # parser.add_argument("--degree", "-d", type=int)
        # parser.add_argument("--column", "-c", default="I0")
        # parser.add_argument("--pvalue", "-p", type=float, default=3)
        subparsers = parser.add_subparsers(dest="finder")
        force = subparsers.add_parser("force")

        variance = subparsers.add_parser("variance")
        variance.add_argument("--width", "-w", type=int, default=10)
        variance.add_argument("--pvalue", "-p", type=float, default=0.001)
        
        smooth = subparsers.add_parser("smooth")
        smooth.add_argument("--fraction", "-f", type=float, default=0.3)
        smooth.add_argument("--width", "-w", type=int, default=10)
        smooth.add_argument("--pvalue", "-p", type=float, default=0.001)
        
        polynomial = subparsers.add_parser("polynomial")
        polynomial.add_argument("--degree", "-d", type=int, default=1)
        polynomial.add_argument("--width", "-w", type=int, default=10)
        polynomial.add_argument("--pvalue", "-p", type=float, default=0.001)
        
        even = subparsers.add_parser("even")
        even.add_argument("--pvalue", "-p", type=float, default=0.001)

        for sp in subparsers._name_parser_map.values():
            subparsers = sp.add_subparsers(dest="method")
            remove = subparsers.add_parser("remove")
            remove.add_argument("--noise", "-n", action="store_true")

            base = subparsers.add_parser("base")
            base.add_argument("--noise", "-n", action="store_true")

            smooth = subparsers.add_parser("smooth")
            smooth.add_argument("--noise", "-n", action="store_true")
            smooth.add_argument("--fraction", "-f", type=float, default=0.05)

        args = parser.parse(tokens)

        _range = parse_range(*args.range)
        if _range.domain == None:
            _range = NumberUnitRange(_range.lower, _range.upper, _range.inter, Domain.REAL)
        if _range.domain != Domain.REAL:
            raise ValueError("Cannot deglitch on non-real domain.")

        match args.finder:
            case "force":
                finder = Finder_Force()
            case "variance":
                finder = Finder_Variance(args.width, args.pvalue)
            case "smooth":
                finder = Finder_Smooth(args.width, args.fraction, args.pvalue)
            case "polynomial":
                finder = Finder_Polynomial(args.width, args.degree, args.pvalue)
            case "even":
                finder = Finder_Even(args.pvalue)
            case f:
                raise ValueError(f"Invalid finder: {f}")
        
        match args.method:
            case "remove":
                method = Method_Remove(args.noise)
            case "base":
                if args.column != "a":
                    raise ValueError("Method base can only be used if column is a.")
                method = Method_Base(args.noise)
            case "smooth":
                method = Method_Smooth(args.noise, args.fraction)
            case f:
                raise ValueError(f"Invalid method: {f}")

        return Args_Deglitch(
            _range,
            args.column,
            finder,
            method
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
            b:npt.NDArray[np.floating] | None = None
            std = None
            
            match args.finder:
                case Finder_Force():
                    g[:] = idx
                case Finder_Variance(width, pvalue):
                    # Estimate the std as median of stds
                    std = pd.Series(i,x).rolling(len(X) // width, center=True).std().median()
                    u = norm.ppf(1-pvalue/2) * std

                    g[idx] = np.abs(i) > u
                case Finder_Smooth(width, fraction, pvalue):
                    # Assume I0 is represented by a low frequency trend
                    #    I0 = I0' + eps
                    # Model I0' as a severely smoothed I0
                    b = lowess(i, x, fraction, it=0, return_sorted=False)
                    y = i-b

                    # Estimate the std as median of stds
                    std = pd.Series(y,x).rolling(len(X) // width, center=True).std().median()
                    u = norm.ppf(1-pvalue/2) * std
                    g[idx] = np.abs(y) > u
                    pass
                case Finder_Polynomial(width, degree, pvalue):
                    # Assume I0 is represented by:
                    #    I0 = P_n + eps
                    # Model the I0 trend as a n-polynomial
                    poly = np.poly1d(np.polyfit(x,i, degree))
                    b = poly(x)
                    y = i-b
                    # Estimate the std as median of stds
                    std = pd.Series(y,x).rolling(len(X) // width, center=True).std().median()
                    u = norm.ppf(1-pvalue/2) * std

                    g[idx] = np.abs(y) > u
                    pass
                case Finder_Even(pvalue):
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
                    b = ii
                    # Estimate the std as median of stds
                    std = pd.Series(y,x).rolling(len(X) // 10, center=True).std().median()
                    u = norm.ppf(1-pvalue/2) * std
                    g[idx] = np.abs(y) > u
            
            match (args.method, b, std):
                case Method_Remove(), _, _:
                    data.datums[domain].df = data.datums[domain].df[idx]
                case (Method_Base(noise), b, std):
                    if b is None or std is None:
                        raise RuntimeError("Method base requires estimation of the baseline and standard deviation.")
                    if args.column != "a":
                        raise RuntimeError("Method base can only be used if the column is a")
                    newcol = I.copy()
                    if noise: n = np.random.normal(0, std, g.sum())
                    else: n = np.zeros(g.sum())
                    
                    newcol[g] = b[g[idx]]+n
                    data.mod_col("a", newcol)
                    pass
                case (Method_Smooth(noise, fraction), _, _):
                    x = data.get_col_("E", domain=domain)
                    y = data.get_col_("a", domain=domain)
                    s = lowess(y[~g],x[~g], it=0, frac=fraction, return_sorted=False, xvals=x)
                    std = (y-s)[idx].std()
                    if noise: n = np.random.normal(0, std, g.sum())
                    else: n = np.zeros(g.sum())
                    
                    yn = y.copy()
                    yn[g] = s[g]+n
                    pass
            

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError

class ArcTangent(NamedTuple):
    a: float
    b: float
    c: float
    def calc(self, x:npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        s = np.tan(0.4 * np.pi)
        # c is defined to be the shift from b to be 10% - 90%
        return self.a * (np.atan(s * (x-self.b)/abs(self.c)) / np.pi + 0.5)
        # return self.a * (np.atan((x-self.b)/self.c) / np.pi + 0.5) # not reparametrized

class HyperTangent(NamedTuple):
    a: float
    b: float
    c: float
    def calc(self, x:npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        s = np.atanh(0.9)
        return self.a * (np.tanh(s * (x-self.b)/abs(self.c)) + 1) * 0.5


class ErrorFunction(NamedTuple):
    a: float
    b: float
    c: float
    def calc(self, x:npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        from scipy.special import erf, erfinv
        s = erfinv(0.8)
        return self.a * (1 + erf(s * (x-self.b)/abs(self.c))) * 0.5

class Exponential(NamedTuple):
    a: float
    b: float
    c: float
    def calc(self, x:npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        y = np.zeros_like(x)
        s = np.log(0.1)
        i = x > self.b

        if self.c > 0:
            y[i] = self.a * (1 - np.exp(s * (x[i]-self.b)/self.c))
        elif self.c < 0:
            y[i] = self.a
            y[~i] = self.a * (np.exp(s * (x[~i]-self.b)/self.c))
        else:
            y[i] = 1

        return y


class Args_MultiEdge(NamedTuple):
    axis: str
    method: ArcTangent|ErrorFunction|Exponential|HyperTangent

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

        tanhyp = subparsers.add_parser("tanh")
        tanhyp.add_argument("b")
        tanhyp.add_argument("a")
        tanhyp.add_argument("c")

        expfun = subparsers.add_parser("exp")
        expfun.add_argument("b")
        expfun.add_argument("a")
        expfun.add_argument("c")

        args = parser.parse(tokens)

        a = parse_nu(args.a).value
        b = parse_nu(args.b).value
        c = parse_nu(args.c).value

        match args.mode:
            case "atan": mode = ArcTangent(a, b, c)
            case "erf": mode = ErrorFunction(a, b, c)
            case "exp": mode = Exponential(a, b, c)
            case "tanh": mode = HyperTangent(a, b, c)

        axis = args.axis if args.axis is not None else "E"
        
        return Args_MultiEdge(axis,mode)

    @staticmethod
    def execute(args: Args_MultiEdge, context: Context) -> CommandResult:
        log = getLogger("multiedge")

        for data in context.data:
            X = data.get_col_(args.axis, domain=Domain.REAL)
            Y = data.get_col_("a", domain=Domain.REAL)

            E = args.method.calc(X)
            # TODO: idea --iplot interactive plot:
            # Si apre un plot e un terminale interattivo.
            # L'utente può cambiare i parametri della curva partendo da quelli
            # dati nel file, per ottimizzare il residuo, che viene plottato ad
            # ogni cambio. Alla fine, stampa la linea del comando finale.

            data.datums[Domain.REAL].add_col(E, Column(None, None, DataColType.MULTIEDGE), "mult", False)
            data.datums[Domain.REAL].mod_col("a", Y-E)

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError

