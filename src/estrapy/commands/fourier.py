from __future__ import annotations

import numpy as np
import numpy.typing as npt

from enum import Enum
from logging import getLogger
from typing import NamedTuple

from ._context import Context, AxisType, Column, DataColType, FourierType, Domain, range_to_index
from ._handler import CommandHandler, Token, CommandResult
from ._numberunit import NumberUnit, parse_range, NumberUnitRange, actualize_range
from ._parser import CommandParser

class Apodizer(Enum):
    HANN = "hann"
    RECTANGLE = "rect"
    BARTLETT = "bart"
    WELCH = "welc"
    SINE = "sine"
    EXPONENTIAL = "expn"
    GAUSS = "gauss"

class Method(Enum):
    DFT = "dft"
    FINUFT = "finuft"
    FFT = "fft"


class Args_Fourier(NamedTuple):
    bounds: NumberUnitRange
    fbounds: NumberUnitRange
    kweight: float
    apodizer: tuple[Apodizer, float | None]
    wwidth: float
    method: Method

def get_window(_r:npt.NDArray[np.floating], apodizer:Apodizer, parameter: float|None=None, symm:bool=True) -> npt.NDArray[np.floating]:
    r0 = np.clip(_r, 0, None)
    if symm:
        r = np.clip(np.abs(_r), 0, 1)
    else:
        r = np.clip(_r, 0, 1)

    match apodizer:
        case Apodizer.HANN:
            return (np.cos(np.pi * r) + 1) / 2
        case Apodizer.RECTANGLE:
            return (r < 1)*1.0
            # return (r <= 0.5)*1.0
        case Apodizer.BARTLETT:
            return 1 - r
        case Apodizer.WELCH:
            return 1 - r * r
        case Apodizer.SINE:
            return np.cos(np.pi * r / 2)
        case Apodizer.EXPONENTIAL:
            assert parameter is not None
            return np.exp(-r0 * parameter)
        case Apodizer.GAUSS:
            assert parameter is not None
            return np.exp(-r0 * r0 * parameter)
    

def get_flattop_window(
    x: npt.NDArray[np.floating],
    width: np.floating | float,
    apodizer: Apodizer,
    parameter: float | None = None,
    symmetry:int = 0,
    shift: tuple[float,float] | float | None = None
) -> npt.NDArray[np.floating]:
    if width == 0:
        return np.ones_like(x)
    
    width = np.min([(np.max(x) - np.min(x))/2, np.float64(width)])

    if symmetry > 0:
        if shift is None: shift = 0
        _r = np.array((np.zeros_like(x), x + width - x.max() - shift))
    elif symmetry < 0:
        if shift is None: shift = 0
        _r = np.array((np.zeros_like(x), x.min() + shift - x + width))
    else:
        if shift is None: shift = 0,0
        _r = np.array((np.zeros_like(x), x + width - x.max() - shift[1], x.min() + shift[0] - x + width)) # type: ignore
    r = np.max(_r,axis=0,)/ width

    return get_window(r, apodizer, parameter)

def fourier(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating | np.complexfloating],
    r: npt.NDArray[np.floating],
) -> npt.NDArray[np.complexfloating]:
    dx = np.diff(x)

    FT = 2 * np.outer(x, r)
    FTl, FTr = FT[:-1, :], FT[1:, :]

    yl = y[:-1] * dx
    yr = y[1:] * dx

    R: npt.NDArray[np.floating] = yl.dot(np.cos(FTl)) + yr.dot(np.cos(FTr))
    I: npt.NDArray[np.floating] = yl.dot(np.sin(FTl)) + yr.dot(np.sin(FTr))

    return (R + 1.0j * I) / np.sqrt(2 * np.pi)

def bfourier(
    r: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
    x: npt.NDArray[np.floating],
) -> npt.NDArray[np.complexfloating]:
    return fourier(r, y.conj(), x).conj()

def finuft(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.complexfloating],
    r: npt.NDArray[np.floating],
) -> npt.NDArray[np.complexfloating]:
    from finufft import nufft1d3
    return nufft1d3(x, y, 2*r, eps=1e-9)

def fft(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    *,
    correct_r: bool=False
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.complexfloating]]:
    
    # Ensure proper x spacing, below a tolerance of 1e-5
    diff = np.diff(x)
    if np.diff(diff).max()/diff.mean() > 1e-5: raise RuntimeError("X spacing is not consistent.")

    # Automatic 2 zerofilling
    n = int(np.exp2(np.ceil(np.log2(len(x)))+2))

    r = np.fft.rfftfreq(n, diff.mean())
    if correct_r:
        r = r * np.pi
    f = np.fft.rfft(y,n)
    return r,f

class Fourier(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Fourier:
        parser = CommandParser("fourier", description="Performs the discrete Fourier transform.")
        parser.add_argument("range", nargs=2, help="The transformation range.")
        parser.add_argument("outrange", nargs=2, help="The maximum R value and spacing of the DFT.")
        parser.add_argument("--kweight", "-k", type=float, default=0, help="The k-weight to scale the signal by.")
        parser.add_argument("--width", "-w", type=float, default=np.inf, help="The ramp width of the apodizer window.")
        parser.add_argument("--apodizer", "-a", default="hann", help="The window function to apodize the data with.")
        parser.add_argument("--method",default="dft",choices=["dft", "finuft", "fft"])

        args = parser.parse(tokens)

        range = parse_range(*args.range)
        outrn = parse_range("0A", *args.outrange)

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
            case "exp":
                _, p = args.apodizer.split(":", maxsplit=1)
                apd = (Apodizer.EXPONENTIAL, float(p))
            case "gau":
                _, p = args.apodizer.split(":", maxsplit=1)
                apd = (Apodizer.GAUSS, float(p))
        
        match args.method:
            case "dft":
                met = Method.DFT
            case "finuft":
                met = Method.FINUFT
            case "fft":
                met = Method.FFT

        return Args_Fourier(range, outrn, args.kweight, apd, args.width, met)

    @staticmethod
    def execute(args: Args_Fourier, context: Context) -> CommandResult:
        log = getLogger("fourier")

        if args.fbounds.domain != Domain.FOURIER:
            raise RuntimeError("The fourier transform requires valid fourier units.")
        
        match args.fbounds.inter:
            case int(n):
                R = np.linspace(args.fbounds.lower.value, args.fbounds.upper.value, n)
            case NumberUnit(v, _, "A"):
                R = np.arange(args.fbounds.lower.value, args.fbounds.upper.value, v)
            case _: raise RuntimeError("Invalid R axis specification.")
        interval = R[1] - R[0]

        domain = args.bounds.domain or Domain.RECIPROCAL
        if domain != Domain.RECIPROCAL:
            raise RuntimeError("Cannot fit postedge to a different domain.")
        _axes = [data.get_col_(data.datums[domain].default_axis) for data in context.data] # type: ignore
        range = actualize_range(args.bounds, _axes, "eV")

        for data in context.data:

            idx = range_to_index(data, range)

            X = data.get_col_("k")
            Y = data.get_col_("x")
            x,_y = X[idx], Y[idx]

            w = get_flattop_window(x, args.wwidth, *args.apodizer) # type: ignore

            # Check if the data has mean 1. If so, the background was not removed,
            # and is estimated as 1
            kw = np.ones(x.shape, dtype=np.float64) if args.kweight == 0 else x ** args.kweight # type: ignore
            kw: npt.NDArray[np.floating]
            y = w * _y * kw

            # Approximate 1/Nyquist frequency
            _min_dR = 2 * np.diff(x).mean()

            log.debug(f"{data.meta.name}: Performing fourier transform")

            if _min_dR > interval:
                log.warning(
                    f"The required frequency ({1/interval:0.1f}) is smaller than the average Nyquist frequency ({1/_min_dR:0.1f}). The minimum interval is {_min_dR:0.4f}"
                )

            match args.method:
                case Method.DFT:
                    f = fourier(x, y, R) # type: ignore
                case Method.FINUFT:
                    f = finuft(x, y+0j, R) # type: ignore
                case Method.FFT:
                    _R,_f = fft(x, y, correct_r=True)
                    selidx = (_R>=args.fbounds.lower.value)&(_R<=args.fbounds.upper.value)
                    R,f = _R[selidx], _f[selidx]

            W = np.zeros_like(X)
            W[idx] = w

            data.add_col("win", W, Column(None, None, DataColType.WINDOW), Domain.RECIPROCAL)

            data.add_col("R", R, Column("A", None, AxisType.DISTANCE), Domain.FOURIER)
            data.datums[Domain.FOURIER].set_default_axis("R")
            
            data.add_col("f", f, Column(None, None, FourierType.COMPLEX), Domain.FOURIER)

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
