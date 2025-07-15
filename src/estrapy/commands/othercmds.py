from __future__ import annotations

import numpy as np

from logging import getLogger
from typing import NamedTuple
from statsmodels.nonparametric.smoothers_lowess import lowess

from ._context import Context, Domain, range_to_index, DataColType, Column
from ._handler import CommandHandler, Token, CommandResult
from ._numberunit import NumberUnit, parse_range, NumberUnitRange, actualize_range
from ._parser import CommandParser


class Args_Cut(NamedTuple):
    bounds: NumberUnitRange


class Cut(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Cut:
        parser = CommandParser("cut", description="Cuts the data.")
        parser.add_argument("range", nargs=2)

        args = parser.parse(tokens)

        range = parse_range(*args.range)

        return Args_Cut(range)

    @staticmethod
    def execute(args: Args_Cut, context: Context) -> CommandResult:
        log = getLogger("cut")

        domain = args.bounds.domain or Domain.REAL
        _axes = [data.get_col_(data.datums[domain].default_axis) for data in context.data] # type: ignore
        range = actualize_range(args.bounds, _axes, "eV")
        
        for data in context.data:
            idx = range_to_index(data, range)
            data.datums[domain].df = data.datums[domain].df.loc[idx, :]
                
            log.debug(f"{data.meta.name}: Cut data in the range {range.lower.value:0.3f}{range.lower.unit} ~ {range.upper.value:0.3f}{range.upper.unit}") # type: ignore

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError


class Args_Smooth(NamedTuple):
    data: str
    xaxis: str
    window: int


class Smooth(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Smooth:
        parser = CommandParser("smooth", description="Smooths the data.")
        parser.add_argument("data", nargs="?", default="a")
        parser.add_argument("xaxis", nargs="?", default="E")
        parser.add_argument("--window", "-w", default=5, type=int, help="The smoothing window.")

        args = parser.parse(tokens)

        return Args_Smooth(args.data, args.xaxis, args.window)

    @staticmethod
    def execute(args: Args_Smooth, context: Context) -> CommandResult:
        log = getLogger("smooth")

        for data in context.data:
            d = data.get_xy(args.xaxis, args.data)
            _frac = args.window / len(d)

            x,y = d.index.to_numpy(), d.to_numpy()
            sy = lowess(y,x, _frac, 0, return_sorted=False)
            data.mod_col(args.data, sy)
            log.debug(f"{data.meta.name}: smoothed column {args.data} with lowess.")
                    
        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError


class Args_Rebin(NamedTuple):
    region: NumberUnitRange

class Rebin(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Rebin:
        parser = CommandParser("rebin", description="Rebins data.")
        parser.add_argument("region", nargs=2)
        parser.add_argument("interval", nargs="?")
        parser.add_argument("--number", "-n", nargs="?", type=int)

        args = parser.parse(tokens)
        # TODO: range must be inclusive
        # Ensure all points are satisfied
        if args.number is not None:
            region = parse_range(*args.region, number=args.number)
        else:
            if args.interval is None:
                raise ValueError("Interval or number of points must be specified.")
            region = parse_range(*args.region, args.interval)

        return Args_Rebin(region)

    @staticmethod
    def execute(args: Args_Rebin, context: Context) -> CommandResult:
        log = getLogger("rebin")

        domain = args.region.domain
        if domain is None:
            raise RuntimeError("Rebin domain is unspecified.")

        if args.region.lower.value == -np.inf or args.region.upper.value == np.inf:
            raise RuntimeError("Rebin range must be specified. Cannot be \"..\".")
        
        match args.region:
            case NumberUnitRange(_, _, None):
                raise RuntimeError("No interval specified for rebin.")

            case NumberUnitRange(_, _, NumberUnit(_,-1|1,"eV")): axis, unit = "e", "eV"
            case NumberUnitRange(_, _, NumberUnit(_,0,"eV")): axis, unit = "E", "eV"
            case NumberUnitRange(_, _, NumberUnit(_,_,"k")): axis, unit = "k", "k"
            case NumberUnitRange(_, _, NumberUnit(_,_,"A")): axis, unit = "R", "A"

            case NumberUnitRange(NumberUnit(_,-1|1,"eV"), _, _): axis, unit = "e", "eV"
            case NumberUnitRange(NumberUnit(_,0,"eV"), _, _): axis, unit = "E", "eV"
            case NumberUnitRange(NumberUnit(_,_,"k"), _, _): axis, unit = "k", "k"
            case NumberUnitRange(NumberUnit(_,_,"A"), _, _): axis, unit = "R", "A"
            
            case NumberUnitRange(_, NumberUnit(_,-1|1,"eV"), _): axis, unit = "e", "eV"
            case NumberUnitRange(_, NumberUnit(_,0,"eV"), _): axis, unit = "E", "eV"
            case NumberUnitRange(_, NumberUnit(_,_,"k"), _): axis, unit = "k", "k"
            case NumberUnitRange(_, NumberUnit(_,_,"A"), _): axis, unit = "R", "A"

            case _:raise RuntimeError("No unit specified for rebin.")
        
        _axes = [data.get_col_(axis) for data in context.data] # type: ignore
        range = actualize_range(args.region, _axes, unit)
        
        match range.inter:
            case NumberUnit(i, _, _):
                newx = np.arange(range.lower.value, range.upper.value, i)
            case int(n):
                newx = np.linspace(range.lower.value, range.upper.value, n)
            case _: raise RuntimeError("Unknown error. #23794608")
        i = newx[1]-newx[0]
        bins = np.concat([[newx[0]-i/2], newx[:]+i/2])
        
        
        for data in context.data:
            X = data.get_col_(axis)
            dig = np.digitize(X, bins)
            df = data.datums[domain]
            df.df = df.df.groupby(dig).mean().loc[1:len(newx),:]
            df.df.loc[:,axis] = newx[df.df.index-1]
        
        # TODO: log
        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError



class Args_Normalize(NamedTuple):
    var: str

class Normalize(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Normalize:
        parser = CommandParser("normalize", description="Normalizes XAS signal to calculate XANES and EXAFS data.")
        parser.add_argument("var")
        args = parser.parse(tokens)

        return Args_Normalize(args.var)

    @staticmethod
    def execute(args: Args_Normalize, context: Context) -> CommandResult:
        log = getLogger("normalize")

        for data in context.data:
            norm = data.meta.get(args.var)
            a = data.get_col_("a")

            data.add_col("mu", a / norm, Column(None, None, DataColType.MU), Domain.REAL)
            data.add_col("x", a / norm - 1, Column(None, None, DataColType.CHI), Domain.REAL)

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
