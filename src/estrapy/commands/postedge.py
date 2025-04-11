from __future__ import annotations

import numpy as np

from enum import Enum
from typing import NamedTuple
from logging import getLogger


from ._context import Context
from ._format import sup, exp
from ._handler import CommandHandler, Token, CommandResult
from ._misc import parse_numberunit_range, NumberUnit
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
        parser = CommandParser(
            "postedge", description="Removes the background postedge contribution."
        )
        parser.add_argument("range", nargs=2, help="The polynomial fit range.")

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

        a,b = parse_numberunit_range(args.range, ("eV", "k", None), fitaxis)

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
            if "postedge" in data.meta.run:
                raise RuntimeError(
                    f"Postedge was already calculated for {data.meta.name}"
                )

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

            log.debug(f"{data.meta.name}: Fitting postedge of order {args.degree} in the region {lb:0.3f}{args.lowerbound.unit} ~ {ub:0.3f}{args.upperbound.unit}")

            idx = idx_l & idx_u
            match args.fitaxis:
                case "eV":
                    X = data.df.rE
                case "k":
                    X = data.df.k

            x, y = X[idx], data.df.x[idx]
            poly = np.poly1d(np.polyfit(x, y, args.degree))

            data.meta.run["postedge"] = (poly, args.fitaxis, args.action)

            data.df["postedge"] = poly(X)
            log.debug(f"{data.meta.name}: postedge = {" ".join(f"{exp(a)}x{sup(e)}" for e,a in enumerate(poly.coef))}")
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
