from __future__ import annotations

import numpy as np

from typing import NamedTuple
from logging import getLogger

from ._context import Context
from ._format import sup, exp
from ._handler import CommandHandler, Token, CommandResult
from ._misc import parse_numberunit_range, NumberUnit
from ._parser import CommandParser


class Args_PreEdge(NamedTuple):
    lowerbound: NumberUnit
    upperbound: NumberUnit
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

        a,b = parse_numberunit_range(args.range, ("eV", None), "eV")

        return Args_PreEdge(a, b, args.degree)

    @staticmethod
    def execute(args: Args_PreEdge, context: Context) -> CommandResult:
        log = getLogger("preedge")
        for data in context.data:
            if "preedge" in data.meta.run:
                raise RuntimeError(
                    f"Preedge was already calculated for {data.meta.name}"
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
                    idx_l = data.df.E > lb
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
                case _:
                    raise RuntimeError("Invalid state exception: #645651")

            log.debug(f"{data.meta.name}: Fitting preedge of order {args.degree} in the region {lb:0.3f}{args.lowerbound.unit} ~ {ub:0.3f}{args.upperbound.unit}")

            idx = idx_l & idx_u
            X = data.df.rE
            x, y = X[idx], data.df.x[idx]
            poly = np.poly1d(np.polyfit(x, y, args.degree))

            log.debug(f"{data.meta.name}: preedge = {" ".join(f"{exp(a)}x{sup(e)}" for e,a in enumerate(poly.coef))}")

            data.meta.run["preedge"] = poly, "eV"

            data.df["preedge"] = poly(X)
            data.df.x = data.df.x - data.df.preedge

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
