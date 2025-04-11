from __future__ import annotations

import numpy as np
import numpy.typing as npt

from enum import Enum
from typing import NamedTuple, Any
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import minimize_scalar, OptimizeResult, root_scalar, RootResults
from scipy.interpolate import interp1d
from logging import getLogger

from ._context import Context
from ._handler import CommandHandler, Token, CommandResult
from ._misc import parse_edgeenergy, parse_numberunit, E_to_sk, NumberUnit
from ._parser import CommandParser

# TODO:
# Add the possibility to set the E0 in Edge command to be relative to the reference E0 if given as relative (+/-eV)


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


FINAL_METHODS = [
    Operation.MAXIMUM,
    Operation.MINIMUM,
    Operation.ZERO,
    Operation.HALFHEIGHT,
    Operation.AVERAGEINT,
]


class Args_Align(NamedTuple):
    method: list[tuple[Operation, int | None]]
    E0: float

    E0s: float
    dE0: float
    wplot: bool
    rplot: bool


class Args_Edge(NamedTuple):
    method: list[tuple[Operation, int | None]]

    E0s: float | None
    dE0: float
    wplot: bool
    rplot: bool

def find_E0_with_method(
    method: list[tuple[Operation, int | None]],
    _bounds: tuple[float, float],
    _y: npt.NDArray[np.floating],
    _x: npt.NDArray[np.floating],
) -> float:
    h: list[tuple[str, Any]] = [("data", (_y, _x - _bounds[0]))]
    bounds = 0, _bounds[1] - _bounds[0]

    for action, arg in method:
        dtype, data = h[-1]
        match action, dtype:
            case [Operation.CUT, "data"]:
                arg = arg or 0
                y, x = data  # type: ignore
                y: npt.NDArray[np.floating]
                x: npt.NDArray[np.floating]
                assert arg is not None
                d = arg * (bounds[1] - bounds[0])
                a, b = bounds[0] - d, bounds[1] + d

                idx = (x >= a) & (x <= b)
                pad = np.pad(idx, (1, 1))
                idx = idx | pad[2:] | pad[:-2]
                h.append(("data", (y[idx], x[idx])))

            case [Operation.POLYFIT, "data"]:
                if arg is None:
                    continue
                y, x = data  # type: ignore
                y: npt.NDArray[np.floating]
                x: npt.NDArray[np.floating]
                poly = np.poly1d(np.polyfit(x, y, arg))
                h.append(("poly", poly))

            case [Operation.SMOOTH, "data"]:
                if arg is None:
                    continue
                y, x = data  # type: ignore
                y: npt.NDArray[np.floating]
                x: npt.NDArray[np.floating]
                y = lowess(y, x, arg / len(x), it=0, return_sorted=False)
                h.append(("data", (y, x)))

            case [Operation.DERIVATIVE, "data"]:
                if arg is None:
                    continue
                y, x = data  # type: ignore
                y: npt.NDArray[np.floating]
                x: npt.NDArray[np.floating]
                for _ in range(arg):
                    y = np.gradient(y, x)
                h.append(("data", (y, x)))

            case [Operation.DERIVATIVE, "poly"]:
                if arg is None:
                    continue
                poly = data
                poly: np.poly1d
                for _ in range(arg):
                    poly = poly.deriv()
                h.append(("poly", poly))

            case [Operation.INTERPOLATE, "data"]:
                if arg is None:
                    continue
                y, x = data  # type: ignore
                y: npt.NDArray[np.floating]
                x: npt.NDArray[np.floating]
                h.append(("interp", interp1d(x, y, arg)))  # type: ignore

            case [Operation.MAXIMUM, "data"]:
                y, x = data  # type: ignore
                y: npt.NDArray[np.floating]
                x: npt.NDArray[np.floating]
                return x[y.argmax()]

            case [Operation.MINIMUM, "data"]:
                y, x = data  # type: ignore
                y: npt.NDArray[np.floating]
                x: npt.NDArray[np.floating]
                return x[y.argmin()]

            case [Operation.MAXIMUM, "poly" | "interp"]:
                p = data
                res: OptimizeResult = minimize_scalar(-p, bounds=bounds)  # type: ignore
                return res.x + _bounds[0]

            case [Operation.MINIMUM, "poly" | "interp"]:
                p = data
                res: OptimizeResult = minimize_scalar(p, bounds=bounds)  # type: ignore
                return res.x + _bounds[0]

            case [Operation.ZERO, "data"]:
                y, x = data  # type: ignore
                y: npt.NDArray[np.floating]
                x: npt.NDArray[np.floating]
                return x[np.abs(y).argmin()] + _bounds[0]

            case [Operation.ZERO, "poly" | "interp"]:
                p = data
                try:
                    res: RootResults = root_scalar(p, bracket=bounds)  # type: ignore
                except ValueError:
                    res: RootResults = root_scalar(p, x0 = bounds[1]/2)  # type: ignore
                return res.root + _bounds[0]

            case [Operation.HALFHEIGHT, "data"]:
                ...
            case [Operation.AVERAGEINT, "data"]:
                ...

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
        case _:
            raise ValueError(f'Invalid method specification: "{method}"')


def parse_method_to_ops(method: str) -> list[tuple[Operation, int | None]]:
    methods = [_parse_method_to_op(m) for m in method.split(".")]
    if any(m[0] in FINAL_METHODS for m in methods[:-1]):
        raise ValueError("Final method in non-final position.")
    if methods[-1][0] not in FINAL_METHODS:
        raise ValueError("Method must terminate with a final method.")

    return methods


class Align(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Align:
        parser = CommandParser(
            "align",
            description="Aligns the reference data to a standard or a reference E0 value.",
        )
        parser.add_argument("method", help="Uses the specified method to find the E0.")
        parser.add_argument(
            "--E0",
            "-E",
            type=parse_edgeenergy,
            required=True,
            help="Sets the E0 value for the edge.",
        )
        parser.add_argument(
            "--dE0", "-d", type=parse_numberunit, help="Interval search width."
        )
        parser.add_argument(
            "--search",
            "-s",
            type=parse_edgeenergy,
            help="Searches around this value. If not set, searches around E0",
        )
        parser.add_argument(
            "--wplot", action="store_true", help="Plots the result of each edge search."
        )
        parser.add_argument(
            "--rplot",
            action="store_true",
            help="Plots the result of all edge searches together.",
        )

        args = parser.parse(tokens)

        E0 = args.E0
        search = args.search if args.search is not None else E0

        if args.dE0 is not None:
            _dE0 = args.dE0
            _dE0: NumberUnit
            if _dE0.unit not in ("eV", None):
                raise ValueError(
                    f'Unit mismatch: unit "{_dE0.unit}" is not compatible with eV.'
                )
            dE0 = _dE0.value
        else:
            dE0 = 5.0

        return Args_Align(
            parse_method_to_ops(args.method),
            E0,
            search,
            dE0,
            args.wplot,
            args.rplot,
        )

    @staticmethod
    def execute(args: Args_Align, context: Context) -> CommandResult:
        log = getLogger("align")
        for data in context.data:
            if data.meta.refE0 is not None:
                raise RuntimeError(
                    f"Reference preedge was already calculated for {data.meta.name}."
                )

            y = data.df.ref.to_numpy()
            x = data.df.E.to_numpy()
            rE0 = find_E0_with_method(
                args.method, (args.E0 - args.dE0, args.E0 + args.dE0), y, x
            )

            shift = rE0 - args.E0

            log.info(
                f"{data.meta.name}: Found reference E0 value at {rE0:.3f}eV (shift by {-shift:+.3f}eV)"
            )

            data.meta.refE0 = args.E0
            data.df.E = data.df.E - shift

            if data.meta.E0 is not None:
                data.meta.E0 = data.meta.E0 - shift

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError


class Edge(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Edge:
        parser = CommandParser(
            "edgeenergy",
            description="Aligns the reference data to a standard or a reference E0 value.",
        )
        parser.add_argument("method", help="Uses the specified method to find the E0.")
        parser.add_argument(
            "--E0",
            "-E",
            type=parse_edgeenergy,
            help="Sets the E0 value for the edge.",
        )
        parser.add_argument(
            "--dE0", "-d", type=parse_numberunit, help="Interval search width."
        )
        parser.add_argument(
            "--wplot", action="store_true", help="Plots the result of each edge search."
        )
        parser.add_argument(
            "--rplot",
            action="store_true",
            help="Plots the result of all edge searches together.",
        )

        args = parser.parse(tokens)

        E0s = args.E0

        if args.dE0 is not None:
            _dE0 = args.dE0
            _dE0: NumberUnit
            if _dE0.unit not in ("eV", None):
                raise ValueError(
                    f'Unit mismatch: unit "{_dE0.unit}" is not compatible with eV.'
                )
            dE0 = _dE0.value
        else:
            dE0 = 5.0

        return Args_Edge(
            parse_method_to_ops(args.method),
            E0s,
            dE0,
            args.wplot,
            args.rplot,
        )

    @staticmethod
    def execute(args: Args_Edge, context: Context) -> CommandResult:
        log = getLogger("edge")
        for data in context.data:
            if data.meta.E0 is not None:
                raise RuntimeError(
                    f"Reference preedge was already calculated for {data.meta.name}."
                )

            y = data.df.x.to_numpy()
            x = data.df.E.to_numpy()
            E0s = E0s if args.E0s is not None else data.meta.refE0
            if E0s is None:
                raise RuntimeError("E0 search interval was not provided.")
            E0 = find_E0_with_method(
                args.method, (E0s - args.dE0, E0s + args.dE0), y, x
            )

            if data.meta.refE0:
                relativeE0 = E0 - data.meta.refE0
                log.info(
                    f"{data.meta.name}: Found E0 value at {E0:.3f}eV ({relativeE0:+.3f}eV)"
                )
            else:
                log.info(f"{data.meta.name}: Found E0 value at {E0:.3f}eV")

            data.meta.E0 = E0
            data.df["rE"] = data.df.E - E0
            data.df["k"] = E_to_sk(data.df.E.to_numpy(), E0)

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
