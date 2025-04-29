from __future__ import annotations

import numpy as np

from enum import Enum
from typing import NamedTuple
from logging import getLogger


from ._context import Context, AxisType, DataColType, Column, Domain, range_to_index
from ._format import sup, exp
from ._handler import CommandHandler, Token, CommandResult
from ._numberunit import NumberUnit, parse_range, NumberUnitRange, actualize_range

from ._parser import CommandParser


class RemovalOperation(Enum):
    DIVIDE = "/"
    SUBTRACT = "-"


class Args_PostEdge(NamedTuple):
    bounds: NumberUnitRange
    action: RemovalOperation
    fitaxis: AxisType
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
            "--energy", "-e", dest="fitaxis", action="store_const", const="e"
        )
        groupa.add_argument(
            "--wavevector", "-k", dest="fitaxis", action="store_const", const="k"
        )

        args = parser.parse(tokens)

        range = parse_range(*args.range)
        if range.domain != Domain.REAL:
            raise ValueError("Invalid fit domain: the postedge can only be calculated in energy or wavevector.")


        # Match bounds and fitaxis
        match args.fitaxis:
            case "e": fitaxis = AxisType.RELENERGY
            case "k": fitaxis = AxisType.KVECTOR
            case None:
                match range.lower:
                    case NumberUnit(_, _, "eV"): fitaxis = AxisType.RELENERGY
                    case NumberUnit(_, _, "k"): fitaxis = AxisType.KVECTOR
                    case _:
                        match range.upper:
                            case NumberUnit(_, _, "eV"): fitaxis = AxisType.RELENERGY
                            case NumberUnit(_, _, "k"): fitaxis = AxisType.KVECTOR
                            case _:
                                raise ValueError("Neither bound has units specified. Specify the fit axis explicitly.")

        match args.method:
            case "div":
                action = RemovalOperation.DIVIDE
            case "sub":
                action = RemovalOperation.SUBTRACT
            case _:
                raise RuntimeError("Invalid removal operation: 458314")
            
        

        return Args_PostEdge(range, action, fitaxis, args.degree)

    @staticmethod
    def execute(args: Args_PostEdge, context: Context) -> CommandResult:
        log = getLogger("postedge")

        domain = args.bounds.domain or Domain.REAL
        if domain != Domain.REAL:
            raise RuntimeError("Cannot fit postedge to a different domain.")
        
        _axes = [data.get_col_(data.datums[domain].default_axis) for data in context.data] # type: ignore
        range = actualize_range(args.bounds, _axes, "eV")

        for data in context.data:
            if "postedge" in data.meta.run:
                raise RuntimeError(f"Postedge was already calculated for {data.meta.name}")
            idx = range_to_index(data, range)
            log.debug(f"{data.meta.name}: Fitting postedge of order {args.degree} in the region {range.lower.value:0.3f}{range.lower.unit} ~ {range.upper.value:0.3f}{range.upper.unit}") # type: ignore

            X = data.get_col_(coltype=args.fitaxis)
            Y = data.get_col_("x")
            x, y = X[idx], Y[idx]
            poly = np.poly1d(np.polyfit(x, y, args.degree)) # type: ignore
            P = poly(X)

            log.debug(f"{data.meta.name}: postedge = {" ".join(f"{exp(a)}x{sup(e)}" for e,a in enumerate(poly.coef))}")
            
            data.meta.run["postedge"] = (poly, args.fitaxis, args.action)
            J0 = poly(0)
            data.meta.run["J0"] = J0
            data.add_col("post", P, Column(None, None, DataColType.POSTEDGE), Domain.REAL)

            match args.action:
                case RemovalOperation.SUBTRACT: Y1 = (Y - P) / J0
                case RemovalOperation.DIVIDE: Y1 = Y / P
            
            data.mod_col("x", Y1)

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
