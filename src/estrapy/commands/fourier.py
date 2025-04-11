from __future__ import annotations

import numpy as np
import numpy.typing as npt

from enum import Enum
from logging import getLogger
from typing import NamedTuple
from matplotlib import pyplot as plt

from ._context import Context
from ._handler import CommandHandler, Token, CommandResult
from ._misc import parse_numberunit_range, parse_numberunit, NumberUnit
from ._parser import CommandParser

class Apodizer(Enum):
    HANN = "hann"
    RECTANGLE = "rect"
    BARTLETT = "bart"
    WELCH = "welc"
    SINE = "sine"

class Args_Fourier(NamedTuple):
    lowerbound: NumberUnit
    upperbound: NumberUnit
    upperdist: float
    interval: float
    kweight: float
    apodizer: tuple[Apodizer, float|None]
    wwidth: float


def get_window(x:npt.NDArray[np.floating], width:float, apodizer:Apodizer, parameter:float|None=None) -> npt.NDArray[np.floating]:
    if width == 0:
        return np.ones_like(x)
    
    r = np.max(np.array((np.zeros_like(x),x+width-x.max(),x.min()-x+width)), axis=0) / width
    match apodizer:
        case Apodizer.HANN:
            return (np.cos(np.pi * r) + 1)/2
        case Apodizer.RECTANGLE:
            return r <= 0.5
        case Apodizer.BARTLETT:
            return 1-r
        case Apodizer.WELCH:
            return 1-r*r
        case Apodizer.SINE:
            return np.cos(np.pi * r / 2)

def fourier(x:npt.NDArray[np.floating],y:npt.NDArray[np.floating],r:npt.NDArray[np.floating]) -> npt.NDArray[np.complexfloating]:
    dx = np.diff(x)

    FT = 2*np.outer(x,r)
    FTl, FTr = FT[:-1,:], FT[1:,:]

    yl = y[:-1]*dx
    yr = y[1:] *dx

    R:npt.NDArray[np.floating] =-yl.dot(np.cos(FTl)) - yr.dot(np.cos(FTr))
    I:npt.NDArray[np.floating] = yl.dot(np.sin(FTl)) + yr.dot(np.sin(FTr))

    return (R + 1.0j*I) / np.sqrt(2*np.pi)

class Fourier(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Fourier:
        parser = CommandParser(
            "fourier", description="Performs the discrete Fourier transform."
        )
        parser.add_argument("range", nargs=2, help="The transformation range.")
        parser.add_argument("out", nargs=2, help="The maximum R value and spacing of the DFT.")
        parser.add_argument("--kweight", "-k", type=float, default=0, help="The k-weight to scale the signal by.")
        parser.add_argument("--width", "-w", type=float, default=0, help="The ramp width of the apodizer window.")
        parser.add_argument("--apodizer", "-a", default="hann", help="The window function to apodize the data with.")

        args = parser.parse(tokens)

        a,b = parse_numberunit_range(args.range, ("eV", "k", None), "k")
        
        R,I = args.out
        _R = parse_numberunit(R)
        if _R.unit not in ("A", None):
            raise ValueError(f"Invalid upper bound specifier: {R}.")
        r = _R.value
        
        _I = parse_numberunit(I)
        if _I.unit not in ("A", None):
            raise ValueError(f"Invalid upper bound specifier: {I}.")
        i = _I.value

        match args.apodizer[:3]:
            case "han":
                apd = (Apodizer.HANN, None)
            case "sin":
                apd = (Apodizer.SINE, None)
            case "rec":
                apd = (Apodizer.RECTANGLE, None)
            case "bar" | "tri":
                apd = (Apodizer.BARTLETT, None)
            case "wel":
                apd = (Apodizer.WELCH, None)

        return Args_Fourier(a, b, r, i, args.kweight, apd, args.width)

    @staticmethod
    def execute(args: Args_Fourier, context: Context) -> CommandResult:
        log = getLogger("fourier")

        R = np.arange(0, args.upperdist, args.interval)
        
        for data in context.data:            
            match args.lowerbound:
                case NumberUnit(value, 0, "eV"):
                    lb = value
                    idx_l = data.df.E >= lb
                case NumberUnit(value, 1 | -1, "eV"):
                    if data.meta.E0 is None:
                        raise RuntimeError(
                            "Cannot specify relative energy value if E0 is not set."
                        )
                    lb = data.meta.E0 + value
                    idx_l = data.df.E >= lb
                case NumberUnit(value, 0, "k"):
                    lb = value
                    idx_l = data.df.k >= lb
                case _:
                    raise RuntimeError("Invalid state exception: #645651")
                
            match args.upperbound:
                case NumberUnit(value, 0, "eV"):
                    ub = value
                    idx_u = data.df.E <= ub
                case NumberUnit(value, 1 | -1, "eV"):
                    if data.meta.E0 is None:
                        raise RuntimeError(
                            "Cannot specify relative energy value if E0 is not set."
                        )
                    ub = data.meta.E0 + value
                    idx_u = data.df.E <= ub
                case NumberUnit(value, 0, "k"):
                    ub = value
                    idx_u = data.df.k <= ub
                case _:
                    raise RuntimeError("Invalid state exception: #645651")

            idx = idx_l & idx_u
            x = data.df.k[idx].to_numpy()
            w = get_window(x, args.wwidth, *args.apodizer)
            y = w * (data.df.x[idx] - 1) * x ** args.kweight

            _min_dR = 2*np.diff(x).mean()

            if _min_dR > args.interval:
                log.warning(f"The required frequency ({1/args.interval:0.2f}) is smaller than the Nyquist frequency ({1/_min_dR:0.2f}). The minimum interval is {_min_dR:0.2f}")

            f = fourier(x,y,R)

            W = np.zeros_like(idx)
            W[idx] = w
            data.df["window"] = W

            data.fd["R"] = R
            data.fd["f"] = f
            
        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
