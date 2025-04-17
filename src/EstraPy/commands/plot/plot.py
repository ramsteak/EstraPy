from __future__ import annotations

import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from typing import NamedTuple, Any
from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib import colors

from ._structs import PlotType, XYPlot, Labels
from .parse_plotkind import get_plot_kind

from .._context import Context
from .._handler import CommandHandler, Token, CommandResult
from .._misc import parse_numberunit_range, parse_numberunit, NumberUnit
from .._parser import CommandParser

QUALITATIVE_CMAPS = {'Pastel1':9,'Pastel2':8,'Paired':12,'Accent':8,'Dark2':8,'Set1':9,'Set2':8,'Set3':12,'tab10':10,'tab20':20,'tab20b':20,'tab20c':20}


class Args_Plot(NamedTuple):
    types: list[PlotType]
    labels: Labels
    xlimits: tuple[float | None, float | None]
    ylimits: tuple[float | None, float | None]
    colorby: str
    colormap: tuple[colors.Colormap, int|None]
    alpha: float
    linewidth: float
    show: bool
    norm: bool


class Plot(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Plot:
        parser = CommandParser("plot", description="Plots the calculated data.")
        parser.add_argument("data", nargs="+")
        parser.add_argument("--xlabel")
        parser.add_argument("--ylabel")
        parser.add_argument("--title")
        parser.add_argument("--xlim", nargs=2)
        parser.add_argument("--ylim", nargs=2)
        parser.add_argument("--colorby")
        parser.add_argument("--color", nargs="+")
        parser.add_argument("--alpha", type=float, default=1.0)
        parser.add_argument("--linewidth", type=float, default=1.0)
        parser.add_argument("--wait", action="store_true")
        parser.add_argument("--normalize", "-n", action="store_true")

        args = parser.parse(tokens)

        # Select plot type from data
        plottypes = [get_plot_kind(d) for d in args.data]

        # Set axis limits
        # I will ignore any unit and there are no defaults
        if args.xlim is not None:
            xl,xu = parse_numberunit_range(args.xlim)
            xl,xu = xl.value, xu.value
            if xl == -np.inf: xl = None
            if xu == np.inf: xu = None
        else: xl,xu = None,None
        if args.ylim is not None:
            yl,yu = parse_numberunit_range(args.ylim)
            yl,yu = yl.value, yu.value
            if yl == -np.inf: yl = None
            if yu == np.inf: yu = None
        else: yl,yu = None,None
        
        colorby = args.colorby if args.colorby is not None else ".n"

        # Select color and colorby
        match args.color:
            case None:
                cmap = colormaps["tab10"]
                cmapcount = QUALITATIVE_CMAPS.get("tab10")
            case [str(_cmapname)] if _cmapname in colormaps:
                cmap = colormaps[_cmapname]
                cmapcount = QUALITATIVE_CMAPS.get(_cmapname)
            case [str(_color)]:
                cmap = colors.LinearSegmentedColormap.from_list("customcmap", [_color])
                cmapcount = None
            case list(_colorlist):
                cmap = colors.LinearSegmentedColormap.from_list('customcmap', _colorlist)
                cmapcount = None
            case _:
                raise RuntimeError()
        
        labels = Labels(
            args.xlabel,# or defaultargs.labels.xlabel,
            args.ylabel,# or defaultargs.labels.ylabel,
            args.title ,#or defaultargs.labels.title,
        )

        return Args_Plot(
            plottypes,
            labels,
            (xl,xu), # type: ignore
            (yl,yu), # type: ignore
            colorby,
            (cmap, cmapcount),
            args.alpha,
            args.linewidth,
            not args.wait,
            args.normalize
        )

    @staticmethod
    def execute(args: Args_Plot, context: Context) -> CommandResult:
        log = getLogger("plot")

        plt.figure()
        for type in args.types:
            match type:
                case XYPlot():
                    if args.colorby is not None:
                        _vals = [data.meta.vars[args.colorby] for data in context.data]

                        if any(isinstance(v, str) for v in _vals):
                            # There is at least one string value, therefore the
                            # color is qualitative and we need to call np.unique
                            ...
                        elif all(isinstance(v, int) for v in _vals):
                            # The variable is discrete, if we use a qualitative
                            # colormap we need to modulo the value by the cmap len
                            if args.colormap[1] is None:
                                _colvs = np.array(_vals)
                                colvs = (_colvs - _colvs.min()) / (_colvs.max() - _colvs.min())
                            else:
                                names, _colvs = np.unique(_vals, return_inverse=True)
                                colvs = _colvs % args.colormap[1]
                        else:
                            # Assume the variable is float
                            _colvs = np.array(_vals)
                            colvs = (_colvs - _colvs.min()) / (_colvs.max() - _colvs.min())

                    for colv,data in zip(colvs,context.data):
                        try:
                            s = type.s.calc(data.df)
                        except KeyError:
                            s = type.s.calc(data.fd)
                        
                        color = args.colormap[0](colv)

                        if args.norm:
                            s = s / np.max(np.abs(s))
                        
                        plt.plot(s, color=color, linewidth=args.linewidth, alpha=args.alpha)

        plt.xlabel(args.labels[0])
        plt.ylabel(args.labels[1])
        plt.title(args.labels[2])

        plt.xlim(args.xlimits)
        plt.ylim(args.ylimits)

        if args.show:
            plt.show()

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
