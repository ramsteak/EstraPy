from __future__ import annotations

import numpy as np
import numpy.typing as npt

from enum import Enum
from logging import getLogger
from typing import NamedTuple
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

from ._context import Context
from ._handler import CommandHandler, Token, CommandResult
from ._misc import parse_numberunit_range, parse_numberunit, NumberUnit
from ._parser import CommandParser


class Args_Cut(NamedTuple):
    range: tuple[NumberUnit, NumberUnit]


class Cut(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Cut:
        parser = CommandParser(
            "cut", description="Cuts the data."
        )
        parser.add_argument("range", nargs=2, help="The transformation range.")

        args = parser.parse(tokens)

        range = parse_numberunit_range(args.range, ("eV", "k", "R", None), "eV")
        return Args_Cut(range)

    @staticmethod
    def execute(args: Args_Cut, context: Context) -> CommandResult:
        log = getLogger("cut")

        match args.range:
            case [NumberUnit(a,sa,("eV"|"k") as ua), NumberUnit(b,sb,("eV"|"k") as ub)]:
                for data in context.data:
                    match ua,sa:
                        case "eV", 0: idx_l = data.df.E >= a
                        case "eV", 1|-1: idx_l = data.df.e >= a
                        case "k": idx_l = data.df.k >= a
                        case _: raise RuntimeError("Invalid state")
                    match ub,sb:
                        case "eV", 0: idx_u = data.df.E <= b
                        case "eV", 1|-1: idx_u = data.df.e <= b
                        case "k": idx_u = data.df.k <= b
                        case _: raise RuntimeError("Invalid state")

                    idx = idx_l & idx_u
                    data.df = data.df[idx]
                    log.info(f"{data.meta.name}: cut data in the range {a}{ua} ~ {b}{ub}")
                    
            case [NumberUnit(a,_,"A"), NumberUnit(b,_,"A")]:
                for data in context.data:
                    idx = (data.fd.R >= a)&(data.fd.R <= b)
                    data.fd = data.fd[idx]
                    log.debug(f"{data.meta.name}: cut data in the range {a}A ~ {b}A")
                    

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError


class Args_Smooth(NamedTuple):
    data: str
    window: int


class Smooth(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Smooth:
        parser = CommandParser(
            "smooth", description="Smooths the data."
        )
        parser.add_argument("data", nargs="?", default="x")
        parser.add_argument("--npoints", "-n", default=5, type=int, help="The smoothing window.")

        args = parser.parse(tokens)

        return Args_Smooth(args.data, args.npoints)

    @staticmethod
    def execute(args: Args_Smooth, context: Context) -> CommandResult:
        log = getLogger("smooth")

        for data in context.data:
            if args.data in data.df.columns:
                y,x = data.df[args.data], data.df.E
                _frac = args.window / len(data.df.E)
                data.df[args.data] = lowess(y, x, _frac, 0, return_sorted=False)
            elif args.data in data.fd.columns:
                y,x = data.fd[args.data], data.fd.R
                _frac = args.window / len(data.fd.R)
                data.fd[args.data] = lowess(y, x, _frac, 0, return_sorted=False)
            else:
                log.error(f"{data.meta.name}: Datapoint {args.data} was not found.")

            log.debug(f"{data.meta.name}: smoothed column {args.data} with lowess.")
                    

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
