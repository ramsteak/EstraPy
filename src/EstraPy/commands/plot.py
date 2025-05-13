from __future__ import annotations

import re
import numpy as np
import numpy.typing as npt
import pandas as pd
import threading

from dataclasses import dataclass, field
from functools import partial
from enum import Enum
from logging import getLogger
from typing import NamedTuple, Any, Callable
from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib import colors
from statsmodels.nonparametric.smoothers_lowess import lowess

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ._context import Context, Data, FigureRuntime, FigureSettings
from ._handler import CommandHandler, Token, CommandResult
from ._numberunit import NumberUnit, Bound, parse_nu, parse_range, NumberUnitRange

from ._parser import CommandParser

QUALITATIVE_CMAPS = {'Pastel1':9,'Pastel2':8,'Paired':12,'Accent':8,'Dark2':8,'Set1':9,'Set2':8,'Set3':12,'tab10':10,'tab20':20,'tab20b':20,'tab20c':20}

class Labels(NamedTuple):
    title: str | None
    xlabel: str | None
    ylabel: str | None

class ColorMap(NamedTuple):
    cmap: colors.Colormap
    num: int | None

class Args_Plot(NamedTuple):
    plot: ColKind | None
    figure: FigureSettings
    labels: Labels | None
    xlimits: NumberUnitRange | None
    ylimits: NumberUnitRange | None
    vshift: float
    colorby: str
    colormap: ColorMap
    alpha: float
    linewidth: float
    show: bool

class ColKind(NamedTuple):
    xcol: str | None
    ycol: str
    xop:list[Callable[[npt.NDArray], npt.NDArray]]
    yop:list[Callable[[npt.NDArray, npt.NDArray], npt.NDArray]]
    xdef: str
    ydef: str

def derivative(y:npt.NDArray, x:npt.NDArray, d:int=1) -> npt.NDArray:
    raise NotImplementedError()

def weight(y:npt.NDArray, x:npt.NDArray, d:int=0) -> npt.NDArray:
    if d <= 0: return y
    return y * x ** d

def weight_noavg(y:npt.NDArray, x:npt.NDArray, d:int=0) -> npt.NDArray:
    if d <= 0: return y
    ym = y.mean()
    return (y - ym) * x ** d + ym

def weight_one(y:npt.NDArray, x:npt.NDArray, d:int=0) -> npt.NDArray:
    if d <= 0: return y
    return (y - 1) * x ** d + 1 # type: ignore

def smooth(y:npt.NDArray, x:npt.NDArray, d:int=1) -> npt.NDArray:
    return lowess(y, x, d/len(x), it=0, is_sorted=False, return_sorted=False)

# XY_CLN_KIND = re.compile(r"^([\w.+*(,)-]+)\:([\w.+*(,)-]+)$")
XY_CLN_KIND = re.compile(r"^(?:([\w.+*(,)-]+)\:)?([\w.+*(,)-]+)$")
def parse_column(p:str) -> ColKind:
    m = XY_CLN_KIND.match(p)
    if m is None: raise ValueError(f"Invalid plot type: {p}")

    x_specs, y_specs = m.groups()
    x_ops, y_ops = [],[]

    if x_specs is not None:
        x_spec = x_specs.split(".")
        for xs in reversed(x_spec[:-1]):
            match xs:
                case "r": op = lambda x: np.real(x)
                case "i": op = lambda x: np.imag(x)
                case "a": op = lambda x: np.abs(x)
                case "p": op = lambda x: np.unwrap(np.angle(x))
                case _: raise ValueError(f"Unknown x operation: {xs}")
            x_ops.append(op)
        xcol = x_spec[-1]
    else:
        xcol = None

    y_spec = y_specs.split(".")
    for ys in reversed(y_spec[:-1]):
        match ys[0]:
            case "r": op = lambda y,_: np.real(y)
            case "i": op = lambda y,_: np.imag(y)
            case "a": op = lambda y,_: np.abs(y)
            case "p": op = lambda y,_: np.unwrap(np.angle(y))
            case "s": op = lambda y,x: partial(smooth,       d=int(ys[1:]))(y,x)
            case "d": op = lambda y,x: partial(derivative,   d=int(ys[1:]))(y,x)
            case "k": op = lambda y,x: partial(weight_one,   d=int(ys[1:]))(y,x)
            case "w": op = lambda y,x: partial(weight,       d=int(ys[1:]))(y,x)
            case "W": op = lambda y,x: partial(weight_noavg, d=int(ys[1:]))(y,x)
            case _: raise ValueError(f"Unknown x operation: {ys}")
        y_ops.append(op)
    ycol = y_spec[-1]

    return ColKind(xcol, ycol, x_ops, y_ops, x_specs, y_specs)

def get_column_(col:ColKind, data:Data) -> tuple[npt.NDArray, npt.NDArray]:
    domain = data._get_col_domain(col.ycol)
    x = data.get_col_(col.xcol, domain=domain)
    for op in col.xop: x = op(x)

    y = data.get_col_(col.ycol, domain=domain)
    for op in col.yop: y = op(y,x)

    return x,y

def get_series(col:ColKind, data:Data) -> pd.Series:
    x,y = get_column_(col, data)
    return pd.Series(y, x)


FIGURE_SUBPLOT = re.compile(r"^(\d+)(?::(\d+)\.(\d+))?(;)?$")
class Plot(CommandHandler):
    @staticmethod
    def parse(tokens: list[Token], context: Context) -> Args_Plot:
        parser = CommandParser("plot", description="Plots the calculated data.")
        parser.add_argument("data", nargs="?")
        parser.add_argument("--xlabel")
        parser.add_argument("--ylabel")
        parser.add_argument("--title")
        parser.add_argument("--xlim", nargs=2)
        parser.add_argument("--ylim", nargs=2)
        parser.add_argument("--vshift", type=float, default=0)
        parser.add_argument("--colorby")
        parser.add_argument("--figure", default=None)
        parser.add_argument("--color", nargs="+")
        parser.add_argument("--alpha", type=float, default=1.0)
        parser.add_argument("--linewidth", type=float, default=1.0)
        parser.add_argument("--show", action="store_true")

        

        args = parser.parse(tokens)

        plot = parse_column(args.data) if args.data is not None else None
        
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
        colormap = ColorMap(cmap, cmapcount)
        
        if any(l is not None for l in (args.xlabel, args.ylabel, args.title)):
            labels = Labels(
                args.title,
                args.xlabel,
                args.ylabel,
            )
        else: labels = None


        show = args.show
        if args.figure is None:
            figsettings = FigureSettings(-1, (1,1))
            context.figures.impl_figsettings.append(figsettings)
        else:
            m = FIGURE_SUBPLOT.match(args.figure)
            if m is None:
                raise ValueError(f"Invalid figure specification: \"{args.figure}\"")
            match m.groups()[0:3]:
                case str(_f),None,None:
                    fignum,a,b = int(_f), 1, 1
                case str(_f),None,None:
                    fignum,a,b = int(_f), 1, 1
                case str(_f),str(_a),str(_b):
                    fignum,a,b = int(_f),int(_a),int(_b)
                case u:
                    raise ValueError(f"Invalid figure specification: \"{u}\"")
            
            if not show and m.group(4) == ";": show = True
            
            figure = FigureSettings(fignum, (a,b))

            if fignum not in context.figures.expl_figsettings:
                context.figures.expl_figsettings[fignum] = figure
            else:
                _prev_fig = context.figures.expl_figsettings[fignum]
                # Update the figure size to the maximum subplot count
                at,bt = (max(a, _prev_fig.subplot[0]),max(b, _prev_fig.subplot[1]))
                context.figures.expl_figsettings[fignum] = FigureSettings(fignum, (at,bt))



        return Args_Plot(
            plot,
            figure,
            labels,
            parse_range(*args.xlim) if args.xlim else None,
            parse_range(*args.ylim) if args.ylim else None,
            args.vshift,
            args.colorby if args.colorby is not None else ".n",
            colormap,
            args.alpha,
            args.linewidth,
            show
        )

    @staticmethod
    def execute(args: Args_Plot, context: Context) -> CommandResult:
        log = getLogger("plot")

        figurenum = args.figure.figurenum
        if figurenum == -1:
            # The figure is in impl_figsettings. Get a nonallocated number.
            figurenum = context.figures.get_impl_figurenum()
            figsettings = FigureSettings(figurenum, (1,1))
            figure = FigureRuntime.new(figsettings)
            context.figures.figureruntimes[figurenum] = figure
        else:
            # The figure was explicitly allocated. Get the actual preexisting figure.
            if figurenum not in context.figures.figureruntimes:
                figsettings = context.figures.expl_figsettings[figurenum]
                figure = FigureRuntime.new(figsettings)

                context.figures.figureruntimes[figurenum] = figure
            else:
                figure = context.figures.figureruntimes[figurenum]

        fig, ax = figure.figure, figure.axes[args.figure.subplot]

        if args.plot is not None:
            for i,data in enumerate(context.data):
                x,y = get_column_(args.plot, data)
                ax._lines.append((x,y))
                shift = i * args.vshift
                
                ax.axis.plot(x, y + shift, linewidth=args.linewidth, alpha=args.alpha)

        if args.labels is not None:
            if args.labels.xlabel is not None: ax.axis.set_xlabel(args.labels.xlabel)
            if args.labels.ylabel is not None: ax.axis.set_ylabel(args.labels.ylabel)
            if args.labels.title is not None: ax.axis.set_title(args.labels.title)


        if args.xlimits is not None:
            match args.xlimits.lower:
                case NumberUnit(_l, _, _) if _l != -np.inf:
                    ax.xlimits = (args.xlimits.lower, ax.xlimits[1])
            match args.xlimits.upper:
                case NumberUnit(_u, _, _) if _u != np.inf:
                    ax.xlimits = (ax.xlimits[0], args.xlimits.upper)
            
        if args.ylimits is not None:
            match args.ylimits.lower:
                case NumberUnit(_l, _, _) if _l != -np.inf:
                    ax.ylimits = (args.ylimits.lower, ax.ylimits[1])
            match args.ylimits.upper:
                case NumberUnit(_u, _, _) if _u != np.inf:
                    ax.ylimits = (ax.ylimits[0], args.ylimits.upper)
        
        match ax.xlimits[0]:
            case Bound.EXTERNAL:...
            case Bound.INTERNAL:...
            case NumberUnit(_l):...

        match ax.xlimits[1]:
            case _:...
        
        match ax.ylimits:
            case _:...

        if args.show:
            figure.show()
            # def on_close(event): fig.canvas.stop_event_loop()
            # fig.canvas.mpl_connect('close_event', on_close)
            # fig.show()
            # fig.canvas.start_event_loop(timeout=-1)
            # figure.shown = True

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
