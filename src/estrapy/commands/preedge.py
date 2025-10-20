from __future__ import annotations

import numpy as np

from typing import NamedTuple
from logging import getLogger

from ._context import Context, Column, DataColType, AxisType, Domain, range_to_index
from ._format import pol
from ._handler import CommandHandler, Token, CommandResult
from ._numberunit import parse_range, NumberUnitRange, actualize_range

from ._parser import CommandParser


class Args_PreEdge(NamedTuple):
    bounds: NumberUnitRange
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

        range = parse_range(*args.range)
        if range.domain != Domain.RECIPROCAL:
            raise ValueError("Invalid fit domain: the preedge can only be calculated in energy or wavevector.")

        return Args_PreEdge(range, args.degree)

    @staticmethod
    def execute(args: Args_PreEdge, context: Context) -> CommandResult:
        log = getLogger("preedge")

        domain = args.bounds.domain or Domain.RECIPROCAL
        if domain != Domain.RECIPROCAL:
            raise RuntimeError("Cannot fit preedge to a different domain.")
        
        _axes = [data.get_col_(data.datums[domain].default_axis) for data in context.data] # type: ignore
        range = actualize_range(args.bounds, _axes, "eV")

        for data in context.data:
            if "preedge" in data.meta.run:
                raise RuntimeError(f"Preedge was already calculated for {data.meta.name}")
            idx = range_to_index(data, range)
            log.debug(f"{data.meta.name}: Fitting preedge of order {args.degree} in the region {range.lower.value:0.3f}{range.lower.unit} ~ {range.upper.value:0.3f}{range.upper.unit}") # type: ignore

            X,Y = data.get_col_("e"), data.get_col_("a")
            x, y = X[idx], Y[idx]
            poly = np.poly1d(np.polyfit(x, y, args.degree)) # type: ignore
            P = poly(X)

            log.debug(f"{data.meta.name}: preedge = {pol(poly.coef)}")

            data.meta.run["preedge"] = poly, AxisType.RELENERGY
            data.add_col("pre", P, Column(None, None, DataColType.PREEDGE), Domain.RECIPROCAL)
            data.mod_col("a", Y - P)
            pass

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
