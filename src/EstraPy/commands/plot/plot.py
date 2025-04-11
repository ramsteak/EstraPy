from __future__ import annotations

import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from typing import NamedTuple
from matplotlib import pyplot as plt

from .._context import Context
from .._handler import CommandHandler, Token, CommandResult
from .._misc import parse_numberunit_range, parse_numberunit, NumberUnit
from .._parser import CommandParser

class Labels(NamedTuple):
    xlabel: str
    ylabel: str
    title: str

@dataclass(slots=True, frozen=True)
class PlotType:
    ...

@dataclass(slots=True, frozen=True)
class XYPlot(PlotType):
    X: str
    Y: str

class Args_Plot(NamedTuple):
    type: PlotType
    labels: Labels
    xlimits: tuple[float | None, float | None]
    ylimits: tuple[float | None, float | None]
    

class Plot(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Plot:
        parser = CommandParser(
            "plot", description="Plots the calculated data."
        )
        parser.add_argument("data")
        parser.add_argument("--xlabel")
        parser.add_argument("--ylabel")
        parser.add_argument("--title")
        parser.add_argument("--xlim", nargs=2)
        parser.add_argument("--ylim", nargs=2)

        args = parser.parse(tokens)

        match args.data:
            case "x(E)":
                plottype = XYPlot("E", "x")
                _limits = (None,None,None,None)
                _accept_xunits = ("eV", None)
                _accept_yunits = (None, )
            case "x(rE)":
                plottype = XYPlot("rE", "x")
                _limits = (0,None,None,None)
                _accept_xunits = ("eV", None)
                _accept_yunits = (None, )
            case "x(k)":
                plottype = XYPlot("k", "x")
                _limits = (0,None,None,None)
                _accept_xunits = ("k", None)
                _accept_yunits = (None, )
            case "f(R)":
                plottype = XYPlot("R", "f")
                _limits = (None,None,None,None)
                _accept_xunits = ("A", None)
                _accept_yunits = (None, )
            case t:
                raise RuntimeError(f"Unrecognized plot type: {t}")
        
        if args.xlim is not None:
            _xlim = parse_numberunit_range(args.xlim, _accept_xunits, default_unit=_accept_xunits[0])
            xlim = _xlim[0].value if _xlim[0].value != -np.inf else None, _xlim[1].value if _xlim[1].value != -np.inf else None
        else:
            xlim = _limits[:2]

        if args.ylim is not None:
            _ylim = parse_numberunit_range(args.ylim, _accept_yunits, default_unit=_accept_yunits[0])
            ylim = _ylim[0].value if _ylim[0].value != -np.inf else None, _ylim[1].value if _ylim[1].value != -np.inf else None
        else:
            ylim = _limits[2:]
        
        return Args_Plot(
            plottype,
            Labels(args.xlabel, args.ylabel, args.title),
            xlim, ylim
        )

    @staticmethod
    def execute(args: Args_Plot, context: Context) -> CommandResult:
        log = getLogger("plot")

        plt.figure()
        match args.type:
            case XYPlot(x,y):
                if x in ["E", "rE", "k"]:
                    for data in context.data:
                        plt.plot(data.df[x], data.df[y])
                elif x in ["R"]:
                    for data in context.data:
                        plt.plot(data.fd[x], data.fd[y])
                else:
                    raise RuntimeError(f"Invalid plot type: {y}({x})")
                
        plt.xlabel(args.labels[0])
        plt.ylabel(args.labels[1])
        plt.title(args.labels[2])
        
        plt.xlim(args.xlimits)
        plt.ylim(args.ylimits)
                    
        plt.show()
            
        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
