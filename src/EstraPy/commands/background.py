from __future__ import annotations

import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from typing import NamedTuple
from matplotlib import pyplot as plt
from scipy import interpolate

from ._context import Context
from ._handler import CommandHandler, Token, CommandResult
from ._misc import parse_numberunit_range, parse_numberunit, NumberUnit
from ._parser import CommandParser

from .fourier import fourier, get_window, Apodizer, get_flattop_window, bfourier


@dataclass(slots=True, frozen=True)
class Method:
    kweight: float

@dataclass(slots=True, frozen=True)
class Spline(Method):
    ...

@dataclass(slots=True, frozen=True)
class BSpline(Method):
    range: tuple[float,float]

@dataclass(slots=True, frozen=True)
class Fourier(Method):
    Rmax: float
    iter: int

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

        args = parser.parse(tokens)

        match args.method:
            case "spline":
                method = Spline(args.kweight)
            case "constant":
                method = Constant(args.kweight)
            case "bspline":
                range = parse_numberunit_range(args.range, ("k", None), "k")
                method = BSpline(args.kweight, (range[0].value, range[1].value))
            case "fourier":
                rmax = parse_numberunit(args.Rmax, ("A", None), "A")
                method = Fourier(args.kweight, rmax.value, args.iterations)

            

        return Args_Background(method)

    @staticmethod
    def execute(args: Args_Background, context: Context) -> CommandResult:
        log = getLogger("background")

        for data in context.data:
            X,Y = data.df.k.to_numpy(), data.df.x.to_numpy()
            match args.method:
                case Constant():
                    S = np.ones_like(X)
                
                case BSpline(kweight, _range):
                    idx = (X>=_range[0])&(X<=_range[1])
                    x, y = X[idx], Y[idx]

                    s = bspline_method(x,y)
                    S = data.df.x.to_numpy().copy()
                    S[idx] = s
                
                case Fourier(kweight, Rmax, iterations):
                    idx = X>=0
                    _Y = Y[idx]
                    x,_y = X[idx], _Y-1
                    y = _y * x ** kweight
                    eps = 0.3
                    w = get_flattop_window(x, (x.max() - x.min())/8, Apodizer.HANN, None, 0, (-eps, +eps))

                    r = np.linspace(-2*Rmax,2*Rmax, 1001)
                    W = get_flattop_window(r, Rmax/2, Apodizer.HANN, None, 0, (Rmax, -Rmax))

                    bkgs:list[npt.NDArray[np.floating]] = []
                    bkgs.append(np.real(bfourier(r, (fourier(x, (y-sum(bkgs))*w, r))*W, x))/2 / w)
                    
                    for _ in range(iterations):
                        bkgs.append(np.real(bfourier(r, (fourier(x, (y-sum(bkgs))*w, r))*W, x))/2 / w)
                    s = sum(bkgs)

                    S = data.df.x.to_numpy().copy()
                    S[idx] = s / x ** kweight +1

                case _:
                    raise NotImplementedError("Method not implemented")


            data.df["bkg"] = S
            data.df.x = data.df.x - S

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
