from __future__ import annotations

import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from typing import NamedTuple
from matplotlib import pyplot as plt
from scipy import interpolate
from statsmodels.nonparametric.smoothers_lowess import lowess

from ._context import Context, Column, DataColType, AxisType, Domain, range_to_index_
from ._handler import CommandHandler, Token, CommandResult
from ._numberunit import NumberUnit, Bound, parse_nu, parse_range, actualize_range

from ._parser import CommandParser

from .fourier import fourier, get_window, Apodizer, get_flattop_window, bfourier, NumberUnitRange

    

@dataclass(slots=True, frozen=True)
class Method:
    kweight: float

@dataclass(slots=True, frozen=True)
class Spline(Method):
    ...

@dataclass(slots=True, frozen=True)
class BSpline(Method):
    range: NumberUnitRange

@dataclass(slots=True, frozen=True)
class Fourier(Method):
    Rmax: float
    iter: int

@dataclass(slots=True, frozen=True)
class Smoothing(Method):
    range: NumberUnitRange
    fraction: float
    iterations: int

@dataclass(slots=True, frozen=True)
class Constant(Method):
    ...

class Args_Background(NamedTuple):
    method:Method

def spline_method(x:npt.NDArray[np.floating], y:npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    ...

def bspline_method(x:npt.NDArray[np.floating], y:npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    s = interpolate.UnivariateSpline(x,y,w=None,bbox=(None,None), k=3, s=None)
    return np.array(s(x))
    

class Background(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Background:
        parser = CommandParser(
            "background",
            description="Calculates the free electron component of the signal.",
        )
        subparsers = parser.add_subparsers(dest="method")

        const = subparsers.add_parser("constant")
        spline = subparsers.add_parser("spline")
        spline.add_argument("--kweight", "-k", type=float)

        bspline = subparsers.add_parser("bspline")
        bspline.add_argument("range", nargs=2)
        bspline.add_argument("--kweight", "-k", default=0, type=float)

        fourier = subparsers.add_parser("fourier")
        fourier.add_argument("Rmax")
        fourier.add_argument("--kweight", "-k", type=float)
        fourier.add_argument("--iterations", "-i", type=int, default=3)

        smooth = subparsers.add_parser("smoothing")
        smooth.add_argument("range", nargs=2)
        smooth.add_argument("--kweight", "-k", type=float)
        smooth.add_argument("--fraction", "-f", type=float, default=0.3)
        smooth.add_argument("--iterations", "-i", type=int, default=1)


        args = parser.parse(tokens)

        match args.method:
            case "spline":
                method = Spline(args.kweight)
            case "constant":
                method = Constant(args.kweight)
            case "bspline":
                range = parse_range(*args.range)
                if range.domain != Domain.REAL:
                    raise ValueError("Background must be fitted to the real domain.")
                method = BSpline(args.kweight, range)
            case "fourier":
                rmax = parse_nu(args.Rmax)
                if rmax.unit != "A": raise ValueError("The max distance must be in angstrom.")
                method = Fourier(args.kweight, rmax.value, args.iterations)
            case "smoothing":
                range = parse_range(*args.range)
                if range.domain != Domain.REAL:
                    raise ValueError("Background must be fitted to the real domain.")
                method = Smoothing(args.kweight, range, args.fraction, args.iterations)

        return Args_Background(method)

    @staticmethod
    def execute(args: Args_Background, context: Context) -> CommandResult:
        log = getLogger("background")

        match args.method:
            case Constant():
                for data in context.data:
                    X,Y = data.get_col_("k", domain=Domain.REAL), data.get_col_("x", domain=Domain.REAL)
                    S = np.ones_like(X)
                    data.add_col("bkg", S, Column(None, None, DataColType.BACKGROUND), Domain.REAL)
                    data.mod_col("x", Y-S)
            
            case BSpline(kweight, _range):
                domain = _range.domain or Domain.REAL
                if domain != Domain.REAL:
                    raise RuntimeError("Cannot fit postedge to a different domain.")
        
                _axes = [data.get_col_("k") for data in context.data]
                bounds = actualize_range(_range, _axes, "k")

                for data in context.data:
                    X,Y = data.get_col_("k", domain=domain), data.get_col_("x", domain=domain)
                    idx = range_to_index_(data, bounds)
                    x, _y = X[idx], Y[idx] -1
                    y = _y * x ** kweight

                    s = bspline_method(x,y)
                    S = Y.copy()
                    S[idx] = s / x ** kweight +1

                    data.add_col("bkg", S, Column(None, None, DataColType.BACKGROUND), Domain.REAL)
                    data.mod_col("x", Y-S)
            
            case Fourier(kweight, Rmax, iterations):
                for data in context.data:
                    X,Y = data.get_col_("k", domain=Domain.REAL), data.get_col_("x", domain=Domain.REAL)
                    idx = X>=0
                    x, _y = X[idx], Y[idx] -1
                    y = _y * x ** kweight
                    eps = 0.3
                    w = get_flattop_window(x, (x.max() - x.min())/8, Apodizer.HANN, None, 0, (-eps, +eps)) # type: ignore

                    r = np.linspace(-2*Rmax,2*Rmax, 1001)
                    W = get_flattop_window(r, Rmax/2, Apodizer.HANN, None, 0, (Rmax, -Rmax))

                    bkgs:list[npt.NDArray[np.floating]] = []
                    bkgs.append(np.real(bfourier(r, (fourier(x, (y-sum(bkgs))*w, r))*W, x))/2 / w) # type: ignore
                    
                    for _ in range(iterations):
                        bkgs.append(np.real(bfourier(r, (fourier(x, (y-sum(bkgs))*w, r))*W, x))/2 / w) # type: ignore
                    s = sum(bkgs)

                    S = Y.copy()
                    S[idx] = s / x ** kweight +1

                    data.add_col("bkg", S, Column(None, None, DataColType.BACKGROUND), Domain.REAL)
                    data.mod_col("x", Y-S)
            
            case Smoothing(kweight, _range, fraction, iterations):
                domain = _range.domain or Domain.REAL
                if domain != Domain.REAL:
                    raise RuntimeError("Cannot fit postedge to a different domain.")
        
                _axes = [data.get_col_("k") for data in context.data]
                bounds = actualize_range(_range, _axes, "k")

                for data in context.data:
                    X,Y = data.get_col_("k", domain=domain), data.get_col_("x", domain=domain)
                    idx = range_to_index_(data, bounds)
                    x,_y = X[idx], Y[idx] -1
                    y = _y * x ** kweight

                    s = lowess(y, x, fraction, iterations, return_sorted=False)
                    S = Y.copy()
                    S[idx] = s / x ** kweight +1
                    data.add_col("bkg", S, Column(None, None, DataColType.BACKGROUND), Domain.REAL)
                    data.mod_col("x", Y-S)
            case _:
                raise NotImplementedError("Method not implemented")

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
