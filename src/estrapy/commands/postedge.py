from __future__ import annotations

import numpy as np

from enum import Enum
from typing import NamedTuple
from logging import getLogger


from ._context import Context, AxisType, DataColType, Column, Domain
from ._format import sup, exp
from ._handler import CommandHandler, Token, CommandResult
from ._misc import parse_numberunit_range, NumberUnit, Bound, parse_numberunit_bound, actualize_bounds
from ._parser import CommandParser


class RemovalOperation(Enum):
    DIVIDE = "/"
    SUBTRACT = "-"


class Args_PostEdge(NamedTuple):
    bounds: tuple[NumberUnit | Bound, NumberUnit | Bound]
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
            "--energy", "-e", dest="fitaxis", action="store_const", const="eV"
        )
        groupa.add_argument(
            "--wavevector", "-k", dest="fitaxis", action="store_const", const="k"
        )

        args = parser.parse(tokens)

        # Match bounds and fitaxis
        match args.fitaxis:
            case "eV":
                fitaxis = AxisType.RELENERGY
                lower,upper = parse_numberunit_bound(args.range, ("eV", "k", None), "eV")
            case "k":
                fitaxis = AxisType.KVECTOR
                lower,upper = parse_numberunit_bound(args.range, ("eV", "k", None), "k")
            case None:
                lower,upper = parse_numberunit_bound(args.range, ("eV", "k", None))
                match lower, upper:
                    case NumberUnit(_, _, None), NumberUnit(_, _, None):
                        raise ValueError("Neither bound has units specified. Specify the fit axis explicitly.")
                    case NumberUnit(_, _, lu), NumberUnit(u, us, None) if lu is not None:
                        match lu:
                            case "eV":
                                fitaxis = AxisType.RELENERGY
                                upper = NumberUnit(u, us, "eV")
                            case "k":
                                fitaxis = AxisType.KVECTOR
                                upper = NumberUnit(u, us, "k")
                            case _:
                                raise RuntimeError("Unknown lower bound unit.")
                    case NumberUnit(l, ls, None), NumberUnit(_, _, uu) if uu is not None:
                        match uu:
                            case "eV":
                                fitaxis = AxisType.RELENERGY
                                lower = NumberUnit(l, ls, "eV")
                            case "k":
                                fitaxis = AxisType.KVECTOR
                                lower = NumberUnit(l, ls, "k")
                            case _:
                                raise RuntimeError("Unknown upper bound unit.")
                    case NumberUnit(_, _, lu), NumberUnit(_, _, uu) if lu == uu:
                        # Both units are the same
                        match lu:
                            case "eV": fitaxis = AxisType.RELENERGY
                            case "k": fitaxis = AxisType.KVECTOR
                            case _: raise RuntimeError("Unknown bounds unit.")
                    case NumberUnit(_, _, lu), NumberUnit(_, _, uu):
                        # Le unità sono diverse
                        raise ValueError("The bounds have different units. Specify the fit axis explicitly.")
                    case _:
                        raise RuntimeError("Unknown error")
            case _:
                raise RuntimeError("Unknown error.")
        
        
        match args.method:
            case "div":
                action = RemovalOperation.DIVIDE
            case "sub":
                action = RemovalOperation.SUBTRACT
            case _:
                raise RuntimeError("Invalid removal operation: 458314")
            
        

        return Args_PostEdge((lower,upper), action, fitaxis, args.degree)

    @staticmethod
    def execute(args: Args_PostEdge, context: Context) -> CommandResult:
        log = getLogger("postedge")

        lower,upper = actualize_bounds(args.bounds, [data.get_col_("E") for data in context.data], "eV")

        for data in context.data:
            if "postedge" in data.meta.run:
                raise RuntimeError(
                    f"Postedge was already calculated for {data.meta.name}"
                )

            match lower:
                case NumberUnit(value, 0, "eV"):
                    idx_l = data.get_col("E") >= value
                case NumberUnit(value, 1 | -1, "eV"):
                    idx_l = data.get_col("e") >= value
                case NumberUnit(value, 0, "k"):
                    idx_l = data.get_col("k") >= value
                case _:
                    raise RuntimeError("Invalid lower bound")
                
            match upper:
                case NumberUnit(value, 0, "eV"):
                    idx_u = data.get_col("E") <= value
                case NumberUnit(value, 1 | -1, "eV"):
                    idx_u = data.get_col("e") <= value
                case NumberUnit(value, 0, "k"):
                    idx_u = data.get_col("k") <= value
                case _:
                    raise RuntimeError("Invalid upper bound")

            log.debug(f"{data.meta.name}: Fitting postedge of order {args.degree} in the region {lower.value:0.3f}{lower.unit} ~ {upper.value:0.3f}{upper.unit}")

            idx = idx_l & idx_u

            X = data.get_col_(coltype=args.fitaxis)
            Y = data.get_col_("x")
            x, y = X[idx], Y[idx]
            poly = np.poly1d(np.polyfit(x, y, args.degree)) # type: ignore
            P = poly(X)

            log.debug(f"{data.meta.name}: postedge = {" ".join(f"{exp(a)}x{sup(e)}" for e,a in enumerate(poly.coef))}")
            
            data.meta.run["postedge"] = (poly, args.fitaxis, args.action)
            data.add_col("post", P, Column(None, DataColType.POSTEDGE), Domain.REAL)

            match args.action:
                case RemovalOperation.SUBTRACT: Y1 = Y - P
                case RemovalOperation.DIVIDE: Y1 = Y / P
            
            data.mod_col("x", Y1)

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
