from __future__ import annotations

import numpy as np
import numpy.typing as npt

# from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt



from enum import Enum
from typing import NamedTuple, Any
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import minimize_scalar, OptimizeResult, root_scalar, RootResults
from logging import getLogger
from dataclasses import dataclass

from ._context import Context, Column, AxisType, FigureSettings, FigureRuntime
from ._handler import CommandHandler, Token, CommandResult
from ._misc import E_to_sk, parse_edgeenergy, nderivative
from ._numberunit import NumberUnit, parse_nu, Domain
from ._parser import CommandParser


class Operation(Enum):
    CUT = "c"

    POLYFIT = "p"
    SMOOTH = "s"
    DERIVATIVE = "d"
    INTERPOLATE = "i"

    MAXIMUM = "M"
    MINIMUM = "m"
    ZERO = "Z"
    HALFHEIGHT = "H"
    AVERAGEINT = "A"
    SET = "S"


FINAL_METHODS = [
    Operation.MAXIMUM,
    Operation.MINIMUM,
    Operation.ZERO,
    Operation.HALFHEIGHT,
    Operation.AVERAGEINT,
    Operation.SET,
]

@dataclass(slots=True, frozen=True)
class Method:...

@dataclass(slots=True, frozen=True)
class Finder(Method):
    operations: list[tuple[Operation, int|None]]
    E0s: float | None
    dE0: float

@dataclass(slots=True, frozen=True)
class Across(Method):
    resolution: int
    dE0: float
    # corrshift: int


class Args_Align(NamedTuple):
    method: Method
    # method: list[tuple[Operation, int | None]]
    E0: float

    wplot: bool


class Args_Edge(NamedTuple):
    method: Method

    wplot: bool

def _get_norm(x:npt.NDArray, shiftcenter:bool=False) -> tuple[float, float]:
    m,M = np.min(x), np.max(x)
    if shiftcenter: return m, M-m
    else:           return 0, max(abs(M), abs(m))

def norm(x:npt.NDArray, center:float, factor:float) -> npt.NDArray:
    return (x-center) / factor

def find_E0_with_method(
    method: list[tuple[Operation, int | None]],
    _bounds: tuple[float, float],
    _y: npt.NDArray[np.floating],
    _x: npt.NDArray[np.floating],
    *, ax:None|Axes=None
) -> float:
    E0shift = (_bounds[1] + _bounds[0])/2

    h: list[tuple[str, Any]] = [("data", (_y, _x - E0shift))]
    _current_bound: tuple[float, float] = np.min(_x) - E0shift, np.max(_x) - E0shift # type: ignore
    _deriv_order = 0
    _norm_factor:dict[int, tuple[float, float]] = {}

    bounds = _bounds[0] - E0shift, _bounds[1] - E0shift

    if ax is not None:
        ax.axvspan(*bounds, alpha=0.1)
        

    for action, arg in method:
        dtype, data = h[-1]

        match action, dtype:
            case [Operation.CUT, "data"]:
                arg = arg or 0
                y, x = data  # type: ignore
                y: npt.NDArray[np.floating]
                x: npt.NDArray[np.floating]
                assert arg is not None
                d = arg * (bounds[1] - bounds[0]) / 2
                a, b = bounds[0] - d, bounds[1] + d

                idx = (x >= a) & (x <= b)
                pad = np.pad(idx, (1, 1))
                idx = idx | pad[2:] | pad[:-2]
                h.append(("data", (y[idx], x[idx])))
                _current_bound = a,b

                if ax:
                    if _deriv_order not in _norm_factor:
                        _norm_factor[_deriv_order] = _get_norm(y[idx], _deriv_order == 0)
                    
                    ax.plot(x[idx], norm(y[idx], *_norm_factor[_deriv_order]), label=f"c{arg}")

            case [Operation.POLYFIT, "data"]:
                if arg is None:
                    continue
                y, x = data  # type: ignore
                y: npt.NDArray[np.floating]
                x: npt.NDArray[np.floating]
                poly = np.poly1d(np.polyfit(x, y, arg))
                h.append(("poly", poly))

                if ax:
                    _pX = np.linspace(*_current_bound, 1001)
                    ax.plot(_pX, norm(poly(_pX), *_norm_factor[_deriv_order]), label=f"p{arg}")

            case [Operation.SMOOTH, "data"]:
                if arg is None:
                    continue
                y, x = data  # type: ignore
                y: npt.NDArray[np.floating]
                x: npt.NDArray[np.floating]
                y = lowess(y, x, arg / len(x), it=0, return_sorted=False)
                h.append(("data", (y, x)))

                if ax:
                    ax.plot(x, norm(y, *_norm_factor[_deriv_order]), label=f"s{arg}")

            case [Operation.DERIVATIVE, "data"]:
                if arg is None:
                    continue
                y, x = data  # type: ignore
                y: npt.NDArray[np.floating]
                x: npt.NDArray[np.floating]
                y = nderivative(y, arg, x) # type: ignore
                _deriv_order += arg
                h.append(("data", (y, x)))

                if ax:
                    if _deriv_order not in _norm_factor:
                        _norm_factor[_deriv_order] = _get_norm(y, _deriv_order == 0)
                    ax.plot(x, norm(y, *_norm_factor[_deriv_order]), label=f"d{arg}")

            case [Operation.DERIVATIVE, "poly"]:
                if arg is None:
                    continue
                poly = data
                poly: np.poly1d
                poly = poly.deriv(arg)
                _deriv_order += arg

                h.append(("poly", poly))

                if ax:
                    _pX = np.linspace(*_current_bound, 1001)
                    if _deriv_order not in _norm_factor:
                        _norm_factor[_deriv_order] = _get_norm(poly(_pX), _deriv_order == 0)

                    ax.plot(_pX, norm(poly(_pX), *_norm_factor[_deriv_order]), label=f"p{arg}")

            case [Operation.INTERPOLATE, "data"]:
                if arg is None:
                    continue
                y, x = data  # type: ignore
                y: npt.NDArray[np.floating]
                x: npt.NDArray[np.floating]
                interp = interp1d(x, y, arg, fill_value=np.nan) # type: ignore
                h.append(("interp", interp))

                if ax:
                    _pX = np.linspace(*_current_bound, 1001)
                    ax.plot(_pX, norm(interp(_pX), *_norm_factor[_deriv_order]), label=f"i{arg}")

            case [Operation.MAXIMUM, "data"]:
                y, x = data  # type: ignore
                y: npt.NDArray[np.floating]
                x: npt.NDArray[np.floating]

                M = x[y.argmax()]
                if ax: ax.axvline(M)
                return M + E0shift

            case [Operation.MINIMUM, "data"]:
                y, x = data  # type: ignore
                y: npt.NDArray[np.floating]
                x: npt.NDArray[np.floating]

                m = x[y.argmin()]
                if ax: ax.axvline(m)
                return m + E0shift

            case [Operation.MAXIMUM, "poly" | "interp"]:
                p = data
                res: OptimizeResult = minimize_scalar(-p, bounds=bounds)  # type: ignore
                if ax: ax.axvline(res.x)
                return res.x + E0shift

            case [Operation.MINIMUM, "poly" | "interp"]:
                p = data
                res: OptimizeResult = minimize_scalar(p, bounds=bounds)  # type: ignore
                if ax: ax.axvline(res.x)
                return res.x + E0shift

            case [Operation.ZERO, "data"]:
                y, x = data  # type: ignore
                y: npt.NDArray[np.floating]
                x: npt.NDArray[np.floating]
                Z = x[np.abs(y).argmin()]
                if ax:
                    ax.axvline(Z)
                    ax.axhline(0, linestyle=":")
                return Z + E0shift

            case [Operation.ZERO, "poly" | "interp"]:
                p = data
                try:
                    res: RootResults = root_scalar(p, bracket=bounds)  # type: ignore
                except ValueError:
                    # Let's try again with the bounded one, but ensuring
                    # that the bracket has different values and includes the
                    # middle value.
                    _b0, _b1 = bounds
                    for i in range(10):
                        if p(_b0) * p(_b1) <= 0: break
                        _b0, _b1 = (_b1-_b0)*0.1, _b1 - (_b1-_b0)*0.1
                    else:
                        raise ValueError("Cannot find root interval.")
                    res: RootResults = root_scalar(p, bracket=(_b0, _b1))  # type: ignore
                if ax:
                    ax.axvline(res.root)
                    ax.axhline(0, linestyle=":")
                return res.root + E0shift

            case [Operation.HALFHEIGHT, "data"]:
                raise NotImplementedError("This operation was not yet implemented.")
            case [Operation.AVERAGEINT, "data"]:
                raise NotImplementedError("This operation was not yet implemented.")
            case [Operation.SET, _]:
                return E0shift
            case [op, dt]:
                raise RuntimeError(f"Invalid operation: {op} on {dt}")
    
    raise ValueError("Method must terminate with a final method.")


def _parse_method_to_op(method: str) -> tuple[Operation, int | None]:
    if len(method) == 1:
        m, a = method, None
    elif len(method) == 2:
        m, a = method[0], int(method[1])
    else:
        raise ValueError(f'Invalid method specification: "{method}"')
    match m:
        case "c":
            return (Operation.CUT, a)
        case "p":
            return (Operation.POLYFIT, a)
        case "s":
            return (Operation.SMOOTH, a)
        case "d":
            return (Operation.DERIVATIVE, a)
        case "i":
            return (Operation.INTERPOLATE, a)

        case "M":
            return (Operation.MAXIMUM, a)
        case "m":
            return (Operation.MINIMUM, a)
        case "Z":
            return (Operation.ZERO, a)
        case "H":
            return (Operation.HALFHEIGHT, a)
        case "A":
            return (Operation.AVERAGEINT, a)
        case "S":
            return (Operation.SET, a)
        case _:
            raise ValueError(f'Invalid method specification: "{method}"')


def parse_method_to_ops(method: str) -> list[tuple[Operation, int | None]]:
    methods = [_parse_method_to_op(m) for m in method.split(".")]
    if any(m[0] in FINAL_METHODS for m in methods[:-1]):
        raise ValueError("Final method in non-final position.")
    if methods[-1][0] not in FINAL_METHODS:
        raise ValueError("Method must terminate with a final method.")

    return methods

_method_aliases = {
    "set": "S",
    "fitderivative": "c1.s5.d1.p3.M",
    "fitpolynomial": "c1.p3.d1.M",
    "fitmaximum": "c1.p3.M",
    "interpderivative": "c1.s5.d1.i3.M",
    "maximum": "c1.i3.M",
}


class Align(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Align:
        parser = CommandParser("align", description="Aligns the reference data to a standard or a reference E0 value.")
        subparser = parser.add_subparsers(dest="method")
        at = subparser.add_parser("at")
        at.add_argument("operations", help="The series of operations to find the E0.")
        at.add_argument("--E0", "-E", type=parse_edgeenergy, required=True, help="Sets the E0 value for the edge.")
        at.add_argument("--dE0", "-d", help="Interval search width.", required=True)
        at.add_argument("--search","-s",type=parse_edgeenergy,help="Searches around this value. If not set, searches around E0")
        at.add_argument("--wplot", action="store_true", help="Plots the result of the edge search.")

        shift = subparser.add_parser("shift")
        shift.add_argument("--dE0", "-d", help="Interval search width.", required=True)
        shift.add_argument("--resolution", default=30, type=int, help="The resolution multiplier.")
        shift.add_argument("--wplot", action="store_true", help="Plots the result of the edge search.")


        # parser.add_argument("method", help="Uses the specified method to find the E0.")
        # parser.add_argument("--E0", "-E", type=parse_edgeenergy, required=True, help="Sets the E0 value for the edge.")
        # parser.add_argument("--dE0", "-d", type=parse_nu, help="Interval search width.")
        # parser.add_argument("--search","-s",type=parse_edgeenergy,help="Searches around this value. If not set, searches around E0")
        # parser.add_argument("--wplot", action="store_true", help="Plots the result of each edge search.")
        # parser.add_argument("--rplot", action="store_true", help="Plots the result of all edge searches together.")
        # parser.add_argument("--resolution", default=30, type=int)


        args = parser.parse(tokens)

        match args.method:
            case "at":
                E0 = args.E0
                search = args.search if args.search is not None else E0
                dE0 = parse_nu(args.dE0)
                if dE0.unit != "eV":
                    raise ValueError(f'Unit mismatch: unit "{dE0.unit}" is not compatible with eV.')
                dE0 = dE0.value

                if args.operations in _method_aliases:
                    ops = _method_aliases[args.operations]
                else: ops = args.operations
                
                method = Finder(parse_method_to_ops(ops), search, dE0)

            case "shift":
                dE0 = parse_nu(args.dE0)
                if dE0.unit != "eV":
                    raise ValueError(f'Unit mismatch: unit "{dE0.unit}" is not compatible with eV.')
                dE0 = dE0.value
                method = Across(args.resolution, dE0)
                E0 = None

            case _: raise ValueError("Unknown method")

        return Args_Align(
            method,
            E0,
            args.wplot,
        )

    @staticmethod
    def execute(args: Args_Align, context: Context) -> CommandResult:
        log = getLogger("align")

        match args.method:
            case Finder(ops, E0s, dE0) if E0s is not None:
                for data in context.data:
                    y = data.get_col_("ref")
                    x = data.get_col_("E")

                    if args.wplot:
                        fignum = context.figures.get_high_figurenum()
                        # TODO: remove FigureRuntime and FigureSettings
                        figure = FigureRuntime.new(FigureSettings(fignum, (1,1)))
                        context.figures.figureruntimes[fignum] = figure
                        ax = figure.axes[(1,1)].axis
                        ax.set_title(f"Reference E0 detection for {data.meta.name}")
                        ax.set_xlabel("Energy shift")
                    else: ax = None

                    try:
                        rE0 = find_E0_with_method(ops, (E0s - dE0, E0s + dE0), y, x, ax=ax)
                    except ValueError:
                        log.error(f"{data.meta.name}: Cannot find reference E0 value with the specified method.")
                        continue
                
                    # Check where the value falls. If it is outside the range, error
                    _l, _u = E0s - dE0, E0s + dE0
                    if (rE0 <= _l) or (rE0 >= _u):
                        log.error(f"{data.meta.name}: E0 value found outside the specified range ({rE0:.3f}eV).")
                        continue
                    elif (rE0 <= _l + 0.05*dE0) or (rE0 >= _u - 0.05*dE0):
                        log.warning(f"{data.meta.name}: E0 value found very close to the specified range ({rE0:.3f}eV)")
                    # If it is in the ±5% of the borders of the range, warning
                
                    if ax is not None:
                        ax.legend()
                        figure.show()

                    shift = rE0 - args.E0

                    log.info(
                        f"{data.meta.name}: Found reference E0 value at {rE0:.3f}eV (shift by {-shift:+.3f}eV)"
                    )

                    data.meta.refE0 = args.E0
                    data.mod_col("E", x - shift)

                    if data.meta.E0 is not None:
                        data.meta.E0 = data.meta.E0 - shift

            case Across(resolution, dE0):
                rXY = [data.get_xy_("E", "ref") for data in context.data]

                # Get common interpolation X axis
                m, M = max(np.min(rX) for rX,_ in rXY), min(np.max(rX) for rX,_ in rXY)
                dx = np.median(np.concat([np.diff(rX) for rX,_ in rXY]))
                newdx = dx/resolution
                cX = np.arange(m, M, newdx)

                # Interpolate the gradients onto the common grid
                drXY = [(rX, np.gradient(rY)) for rX, rY in rXY]
                dYs = [interp1d(rX, drY, "cubic")(cX) / np.std(drY) for rX, drY in drXY]

                # Calculate the point shift from the dE0
                corrshift = int(np.ceil(dE0 / newdx))
                corrs = [np.correlate(dY, dYs[0][corrshift:-corrshift], "valid") for dY in dYs]
                cx = np.linspace(-newdx * corrshift, newdx * corrshift, len(corrs[0]))

                shifts = np.array([np.poly1d(np.polyfit(cx, corr, 2)).deriv().roots[0] for corr in corrs])
                shifts = shifts - np.mean(shifts)

                for data, shift in zip(context.data, shifts):
                    data.mod_col("E", data.get_col_("E") - shift)
                    data.meta.refE0 = args.E0

                if args.wplot:
                    fignum = context.figures.get_high_figurenum()
                    fig = plt.figure(fignum)
                    ax11 = fig.add_subplot(2,2,1)
                    ax11.hist(shifts, bins=30)

                    ax12 = fig.add_subplot(2,2,2)
                    for c in corrs:
                        ax12.plot(cx, c)
                    
                    ax21 = fig.add_subplot(2,2,3)
                    ax21.plot(shifts)

                    ax21 = fig.add_subplot(2,2,4)
                    for rX, rY in rXY:
                        plt.plot(rX, rY, color="blue")
                    for rX, rY in rXY:
                        plt.plot(rX, rY, color="red", alpha=0.5, linewidth=1)
                    

                
        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError


class Edge(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Edge:
        parser = CommandParser("edgeenergy", description="Aligns the reference data to a standard or a reference E0 value.")
        parser.add_argument("operations", help="Uses the specified method to find the E0.")
        parser.add_argument("--E0", "-E", type=parse_edgeenergy, help="Searches around this value. If not set, searches around reference E0.")
        parser.add_argument("--dE0", "-d", type=parse_nu, help="Interval search width.")
        parser.add_argument("--wplot", action="store_true", help="Plots the result of each edge search.")

        args = parser.parse(tokens)

        E0s:float|None = args.E0

        if args.dE0 is not None:
            _dE0 = args.dE0
            _dE0: NumberUnit
            if _dE0.unit != "eV":
                raise ValueError(f'Unit mismatch: unit "{_dE0.unit}" is not compatible with eV.')
            dE0 = _dE0.value
        else:
            dE0 = 5.0

        match args.operations:
            case str(ops):
                if ops in _method_aliases:
                    ops = _method_aliases[ops]
                method = Finder(parse_method_to_ops(ops), E0s, dE0)

        return Args_Edge(
            method,
            args.wplot,
        )

    @staticmethod
    def execute(args: Args_Edge, context: Context) -> CommandResult:
        log = getLogger("edge")

        for data in context.data:
            if data.meta.E0 is not None:
                raise RuntimeError(
                    f"Reference preedge was already calculated for {data.meta.name}."
                )
            
            match args.method:
                case Finder(ops, E0s, dE0):
                    x = data.get_col_("E")
                    y = data.get_col_("a")

                    if E0s is None:
                        E0s = data.meta.refE0
                        if E0s is None:
                            raise RuntimeError("E0 search interval was not provided.")

                    if args.wplot:
                        fignum = context.figures.get_high_figurenum()
                        figure = FigureRuntime.new(FigureSettings(fignum, (1,1)))
                        context.figures.figureruntimes[fignum] = figure
                        ax = figure.axes[(1,1)].axis
                        ax.set_title(f"E0 detection for {data.meta.name}")
                        ax.set_xlabel("Energy shift")
                    else: ax = None

                    try:
                        E0 = find_E0_with_method(ops, (E0s - dE0, E0s + dE0), y, x, ax=ax) # type: ignore
                    except ValueError:
                        log.error(f"{data.meta.name}: Cannot find E0 value with the specified method.")
                        if ax is not None:
                            ax.legend()
                            figure.show()
                        continue

                    # Check where the value falls. If it is outside the range, error
                    _l, _u = E0s - dE0, E0s + dE0
                    if (E0 <= _l) or (E0 >= _u):
                        log.error(f"{data.meta.name}: E0 value found outside the specified range ({E0:.3f}eV).")
                        continue
                    elif (E0 <= _l + 0.05*dE0) or (E0 >= _u - 0.05*dE0):
                        log.warning(f"{data.meta.name}: E0 value found very close to the specified range ({E0:.3f}eV)")
                    # If it is in the ±5% of the borders of the range, warning
                
                    if ax is not None:
                        ax.legend()
                        figure.show()
                    
                    if data.meta.refE0:
                        relativeE0 = E0 - data.meta.refE0
                        log.info(
                            f"{data.meta.name}: Found E0 value at {E0:.3f}eV ({relativeE0:+.3f}eV from reference)"
                        )
                    else:
                        log.info(f"{data.meta.name}: Found E0 value at {E0:.3f}eV")

                    data.meta.E0 = E0
                case _: raise RuntimeError("Invalid state: #23578423")
            
            data.add_col("e", np.array(x - E0), Column("eV", True, AxisType.RELENERGY), Domain.REAL)
            data.add_col("k", E_to_sk(x, E0), Column("k", None, AxisType.KVECTOR), Domain.REAL)

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
