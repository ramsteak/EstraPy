from __future__ import annotations

import numpy as np
import numpy.typing as npt

from enum import Enum
from typing import NamedTuple, Any
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import minimize_scalar, OptimizeResult, root_scalar, RootResults
from scipy.interpolate import interp1d
from logging import getLogger

from matplotlib import pyplot as plt

from ._context import Context
from ._handler import CommandHandler, Token, CommandResult
from ._misc import parse_numberunit, NumberUnit
from ._parser import CommandParser


class RemovalOperation(Enum):
    DIVIDE = "/"
    SUBTRACT = "-"


class Args_PostEdge(NamedTuple):
    lowerbound: NumberUnit
    upperbound: NumberUnit
    action: RemovalOperation
    fitaxis: str
    degree: int


class PostEdge(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_PostEdge:
        # The first two arguments must be the lower and upper bound, and are
        # removed from the token list before parsing
        token_a, token_b, *tokens = tokens
        range = token_a.value, token_b.value

        parser = CommandParser(
            "postedge", description="Removes the background postedge contribution."
        )
        # parser.add_argument("range", nargs=2, help="The polynomial fit range.")

        groupd = parser.add_mutually_exclusive_group()
        groupd.add_argument(
            "--constant",
            "-C",
            dest="degree",
            action="store_const",
            const=0,
            help="Models the postedge as a constant polynomial.",
        )
        groupd.add_argument(
            "--linear",
            "-l",
            dest="degree",
            action="store_const",
            const=1,
            help="Models the postedge as a linear polynomial.",
        )
        groupd.add_argument(
            "--quadratic",
            "-q",
            dest="degree",
            action="store_const",
            const=2,
            help="Models the postedge as a quadratic polynomial.",
        )
        groupd.add_argument(
            "--cubic",
            "-c",
            dest="degree",
            action="store_const",
            const=3,
            help="Models the postedge as a cubic polynomial.",
        )
        groupd.add_argument(
            "--polynomial",
            "-p",
            dest="degree",
            type=int,
            help="Models the postedge as a polynomial trend, and removes it from the data.",
        )

        groupm = parser.add_mutually_exclusive_group()
        groupm.add_argument(
            "--divide", "-d", dest="method", action="store_const", const="div"
        )
        groupm.add_argument(
            "--subtract", "-s", dest="method", action="store_const", const="sub"
        )

        groupa = parser.add_mutually_exclusive_group()
        groupa.add_argument(
            "--energy", "-e", dest="fitaxis", action="store_const", const="eV"
        )
        groupa.add_argument(
            "--wavevector", "-k", dest="fitaxis", action="store_const", const="k"
        )

        args = parser.parse(tokens)

        fitaxis = args.fitaxis if args.fitaxis is not None else "eV"

        match range:
            case [str(A), ".."]:
                b = NumberUnit(np.inf, 0, "eV")

                _a = parse_numberunit(A)
                if _a.unit in ("eV", "k"):
                    a = _a
                elif _a.unit is None:
                    a = NumberUnit(_a.value, _a.sign, fitaxis)
                else:
                    raise ValueError(f"Invalid lower bound specifier: {A}.")

            case [str(A), str(B)]:
                _a = parse_numberunit(A)
                if _a.unit in ("eV", "k"):
                    a = _a
                elif _a.unit is None:
                    a = NumberUnit(_a.value, _a.sign, fitaxis)
                else:
                    raise ValueError(f"Invalid lower bound specifier: {A}.")

                _b = parse_numberunit(B)
                if _b.unit in ("eV", "k"):
                    b = _b
                elif _b.unit is None:
                    b = NumberUnit(_b.value, _b.sign, fitaxis)
                else:
                    raise ValueError(f"Invalid upper bound specifier: {B}.")

            case [A, B]:
                raise ValueError(f"Invalid range specifier: {A} {B}")

        match args.method:
            case "div":
                action = RemovalOperation.DIVIDE
            case "sub":
                action = RemovalOperation.SUBTRACT
            case _:
                raise RuntimeError("Invalid removal operation: 458314")

        return Args_PostEdge(a, b, action, fitaxis, args.degree)

    @staticmethod
    def execute(args: Args_PostEdge, context: Context) -> CommandResult:
        log = getLogger("postedge")
        for data in context.data:
            if "postedge" in data.metadata.run:
                raise RuntimeError(
                    f"Postedge was already calculated for {data.metadata.name}"
                )

            match args.lowerbound:
                case NumberUnit(value, 0, "eV"):
                    lb = value
                    idx_l = data.df.E >= lb
                case NumberUnit(value, 1 | -1, "eV"):
                    if data.metadata.E0 is None:
                        raise RuntimeError(
                            "Cannot specify relative energy value if E0 is not set."
                        )
                    lb = data.metadata.E0 + value
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
                    if data.metadata.E0 is None:
                        raise RuntimeError(
                            "Cannot specify relative energy value if E0 is not set."
                        )
                    ub = data.metadata.E0 + value
                    idx_u = data.df.E <= ub
                case NumberUnit(value, 0, "k"):
                    ub = value
                    idx_l = data.df.k <= ub
                case _:
                    raise RuntimeError("Invalid state exception: #645651")

            log.debug(
                f"Fitting postedge of order {args.degree} in the region {lb:0.3f}{args.lowerbound.unit} ~ {ub:0.3f}{args.upperbound.unit}"
            )

            idx = idx_l & idx_u
            match args.fitaxis:
                case "eV":
                    X = data.df.E
                case "k":
                    X = data.df.k

            x, y = X[idx], data.df.x[idx]
            poly = np.poly1d(np.polyfit(x, y, args.degree))

            data.metadata.run["postedge"] = (poly, args.fitaxis, args.action)

            data.df["postedge"] = poly(X)

            match args.action:
                case RemovalOperation.SUBTRACT:
                    data.df.x = data.df.x - data.df.postedge
                case RemovalOperation.DIVIDE:
                    data.df.x = data.df.x / data.df.postedge

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
