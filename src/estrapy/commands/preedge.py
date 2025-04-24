from __future__ import annotations

import numpy as np

from typing import NamedTuple
from logging import getLogger

from ._context import Context, Column, SignalType, DataColType, AxisType, Domain
from ._format import sup, exp
from ._handler import CommandHandler, Token, CommandResult
from ._misc import parse_numberunit_range, NumberUnit, Bound, parse_numberunit_bound, actualize_bounds
from ._parser import CommandParser


class Args_PreEdge(NamedTuple):
    bounds: tuple[NumberUnit | Bound, NumberUnit | Bound]
    degree: int

class PreEdge(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_PreEdge:
        parser = CommandParser(
            "preedge", description="Removes the background preedge contribution."
        )
        parser.add_argument("range", nargs=2, help="The polynomial fit range.")

        groupd = parser.add_mutually_exclusive_group()
        groupd.add_argument(
            "--constant",
            "-C",
            dest="degree",
            action="store_const",
            const=0,
            help="Models the preedge as a constant polynomial.",
        )
        groupd.add_argument(
            "--linear",
            "-l",
            dest="degree",
            action="store_const",
            const=1,
            help="Models the preedge as a linear polynomial.",
        )
        groupd.add_argument(
            "--quadratic",
            "-q",
            dest="degree",
            action="store_const",
            const=2,
            help="Models the preedge as a quadratic polynomial.",
        )
        groupd.add_argument(
            "--cubic",
            "-c",
            dest="degree",
            action="store_const",
            const=3,
            help="Models the preedge as a cubic polynomial.",
        )
        groupd.add_argument(
            "--polynomial",
            "-p",
            dest="degree",
            type=int,
            help="Models the preedge as a polynomial trend, and removes it from the data.",
        )

        args = parser.parse(tokens)

        lower,upper = parse_numberunit_bound(args.range, ("eV", None), "eV")

        return Args_PreEdge((lower,upper), args.degree)

    @staticmethod
    def execute(args: Args_PreEdge, context: Context) -> CommandResult:
        log = getLogger("preedge")

        lower,upper = actualize_bounds(args.bounds, [data.get_col_("E") for data in context.data], "eV")

        for data in context.data:
            if "preedge" in data.meta.run:
                raise RuntimeError(
                    f"Preedge was already calculated for {data.meta.name}"
                )

            match lower:
                case NumberUnit(value, 0, "eV"):
                    idx_l = data.get_col("E") >= value
                case NumberUnit(value, 1 | -1, "eV"):
                    idx_l = data.get_col("e") >= value
                case _:
                    raise RuntimeError("Invalid state exception: #645651")

            match upper:
                case NumberUnit(value, 0, "eV"):
                    idx_u = data.get_col("E") <= value
                case NumberUnit(value, 1 | -1, "eV"):
                    idx_u = data.get_col("e") <= value
                case _:
                    raise RuntimeError("Invalid state exception: #645651")

            log.debug(f"{data.meta.name}: Fitting preedge of order {args.degree} in the region {lower.value:0.3f}{lower.unit} ~ {upper.value:0.3f}{upper.unit}")

            idx = idx_l & idx_u

            X,Y = data.get_col_("e"), data.get_col_("x")
            x, y = X[idx], Y[idx]
            poly = np.poly1d(np.polyfit(x, y, args.degree)) # type: ignore
            P = poly(X)

            log.debug(f"{data.meta.name}: preedge = {" ".join(f"{exp(a)}x{sup(e)}" for e,a in enumerate(poly.coef))}")

            data.meta.run["preedge"] = poly, AxisType.RELENERGY
            data.add_col("pre", P, Column(None, DataColType.PREEDGE), Domain.REAL)
            data.mod_col("x", Y - P)
            pass

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
