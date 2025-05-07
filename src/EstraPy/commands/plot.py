from __future__ import annotations

import re
import numpy as np
import numpy.typing as npt
import pandas as pd
import threading

from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger
from typing import NamedTuple, Any, Callable
from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib import colors
from statsmodels.nonparametric.smoothers_lowess import lowess

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ._context import Context, Data
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

class FigureSettings(NamedTuple):
    figurenum: int
    subplot: tuple[int,int]

@dataclass(slots=True)
class FigureRuntime:
    settings: FigureSettings
    figure: Figure
    axes: dict[tuple[int,int], AxisRuntime]
    shown: bool

@dataclass(slots=True)
class AxisRuntime:
    axis: Axes
    xmins: list[float] = field(default_factory=list)
    ymins: list[float] = field(default_factory=list)
    xmaxs: list[float] = field(default_factory=list)
    ymaxs: list[float] = field(default_factory=list)



class Args_Plot(NamedTuple):
    plot: ColKind | None
    figure: FigureSettings
    labels: Labels | None
    xlimits: NumberUnitRange | None
    ylimits: NumberUnitRange | None
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
def smooth(y:npt.NDArray, x:npt.NDArray, d:int=1) -> npt.NDArray:
    return lowess(y, x, d/len(x), it=0, is_sorted=False, return_sorted=False)

# XY_CLN_KIND = re.compile(r"^([\w.+*(,)-]+)\:([\w.+*(,)-]+)$")
XY_CLN_KIND = re.compile(r"^(?:([\w.+*(,)-]+)\:)?([\w.+*(,)-]+)$")
def parse_column(p:str) -> ColKind:
    m = XY_CLN_KIND.match(p)
    if m is None: raise ValueError(f"Invalid plot type: {p}")

    Xdef, Ydef = m.groups()
    x_ops, y_ops = [],[]

    if Xdef is not None:
        xs = Xdef.split(".")
        for x in reversed(xs[:-1]):
            match x:
                case "r": op = lambda x: np.real(x)
                case "i": op = lambda x: np.imag(x)
                case "a": op = lambda x: np.abs(x)
                case "p": op = lambda x: np.unwrap(np.angle(x))
                case _: raise ValueError(f"Unknown x operation: {x}")
            x_ops.append(op)
        xcol = xs[-1]
    else:
        xcol = None

    ys = Ydef.split(".")
    for y in reversed(ys[:-1]):
        match y[0]:
            case "r": op = lambda y,_: np.real(y)
            case "i": op = lambda y,_: np.imag(y)
            case "a": op = lambda y,_: np.abs(y)
            case "p": op = lambda y,_: np.unwrap(np.angle(y))
            case "s": op = lambda y,x: smooth(y, x, int(y[1:]))
            case "d": op = lambda y,x: derivative(y,x, int(y[1:]))
            case _: raise ValueError(f"Unknown x operation: {y}")
        y_ops.append(op)
    ycol = ys[-1]

    return ColKind(xcol, ycol, x_ops, y_ops, Xdef, Ydef)

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

        if "figurelist" not in context.options.other:
            context.options.other["figurelist"] = dict[int, FigureSettings]()
        figurelist = context.options.other["figurelist"]
        figurelist:dict[int, FigureSettings]

        show = args.show

        if args.figure is None:
            f = max(figurelist)+1 if figurelist else 1
            figure = FigureSettings(f, (1,1))
            figurelist[f] = figure
        else:
            m = FIGURE_SUBPLOT.match(args.figure)
            if m is None:
                raise ValueError(f"Invalid figure specification: \"{args.figure}\"")
            match m.groups()[0:3]:
                case str(_f),None,None:
                    f,a,b = int(_f), 1, 1
                case str(_f),None,None:
                    f,a,b = int(_f), 1, 1
                case str(_f),str(_a),str(_b):
                    f,a,b = int(_f),int(_a),int(_b)
                case u:
                    raise ValueError(f"Invalid figure specification: \"{u}\"")
            
            if not show and m.group(4) == ";": show = True
            
            figure = FigureSettings(f, (a,b))

            if f not in figurelist:
                figurelist[f] = figure
            else:
                _fig = figurelist[f]
                at,bt = (max(a, _fig.subplot[0]),max(b, _fig.subplot[1]))
                figurelist[f] = FigureSettings(f, (at,bt))



        return Args_Plot(
            plot,
            figure,
            labels,
            parse_range(*args.xlim) if args.xlim else None,
            parse_range(*args.ylim) if args.ylim else None,
            args.colorby if args.colorby is not None else ".n",
            colormap,
            args.alpha,
            args.linewidth,
            show
        )

    @staticmethod
    def execute(args: Args_Plot, context: Context) -> CommandResult:
        log = getLogger("plot")

        if "figurerun" not in context.options.other:
            context.options.other["figurerun"] = dict[int, FigureRuntime]()
        all_figures = context.options.other["figurerun"]
        all_figures:dict[int, FigureRuntime]

        figurelist = context.options.other["figurelist"]
        figurelist:dict[int, FigureSettings]

        if args.figure.figurenum not in all_figures:
            _figsettings = figurelist[args.figure.figurenum]
            _fig = plt.figure(args.figure.figurenum)
            _subplots = figurelist[args.figure.figurenum].subplot
            match _subplots:
                case (1,1):
                    _ax = {(1,1):AxisRuntime(_fig.subplots(1,1))}
                case (1,_):
                    _axs = _fig.subplots(*_subplots)
                    _ax = {(1,c):AxisRuntime(_ax) for c,_ax in enumerate(_axs,1)}
                case (_,1):
                    _axs = _fig.subplots(*_subplots)
                    _ax = {(r,1):AxisRuntime(_ax) for r,_ax in enumerate(_axs,1)}
                case (_,_):
                    _axss = _fig.subplots(*_subplots)
                    _ax = {(r,c):AxisRuntime(_ax) for r,_axs in enumerate(_axss,1) for c,_ax in enumerate(_axs,1)}
                case _: raise RuntimeError("Unknown error: #3409234")
                

            figure = FigureRuntime(_figsettings, _fig, _ax, False)
            all_figures[args.figure.figurenum] = figure
        else:
            figure = all_figures[args.figure.figurenum]

        fig, ax = figure.figure, figure.axes[args.figure.subplot]

        if args.plot is not None:
            for data in context.data:
                x,y = get_column_(args.plot, data)
                ax.xmins.append(np.min(x))
                ax.xmaxs.append(np.max(x))
                ax.ymins.append(np.min(y))
                ax.ymaxs.append(np.max(y))

                ax.axis.plot(x, y, linewidth=args.linewidth, alpha=args.alpha)


        if args.labels is not None:
            if args.labels.xlabel is not None: ax.axis.set_xlabel(args.labels.xlabel)
            if args.labels.ylabel is not None: ax.axis.set_ylabel(args.labels.ylabel)
            if args.labels.title is not None: ax.axis.set_title(args.labels.title)

        if args.xlimits is not None:
            match args.xlimits.lower:
                case Bound.EXTERNAL: x_lower = min(ax.xmins)
                case Bound.INTERNAL: x_lower = max(ax.xmins)
                case NumberUnit(_l,_,_) if _l == -np.inf: x_lower = None
                case NumberUnit(_l,_,_): x_lower = _l
            match args.xlimits.upper:
                case Bound.EXTERNAL: x_upper = max(ax.xmaxs)
                case Bound.INTERNAL: x_upper = min(ax.xmaxs)
                case NumberUnit(_u,_,_) if _u == np.inf: x_upper = None
                case NumberUnit(_u,_,_): x_upper = _u
            ax.axis.set_xlim(x_lower, x_upper)

        if args.ylimits is not None:
            match args.ylimits.lower:
                case Bound.EXTERNAL: y_lower = min(ax.ymins)
                case Bound.INTERNAL: y_lower = max(ax.ymins)
                case NumberUnit(_l,_,_) if _l == -np.inf: y_lower = None
                case NumberUnit(_l,_,_): y_lower = _l
            match args.ylimits.upper:
                case Bound.EXTERNAL: y_upper = max(ax.ymaxs)
                case Bound.INTERNAL: y_upper = min(ax.ymaxs)
                case NumberUnit(_u,_,_) if _u == np.inf: y_upper = None
                case NumberUnit(_u,_,_): y_upper = _u
            ax.axis.set_ylim(y_lower, y_upper)

        if args.show:
            def on_close(event): fig.canvas.stop_event_loop()
            fig.canvas.mpl_connect('close_event', on_close)
            fig.show()
            fig.canvas.start_event_loop(timeout=-1)
            figure.shown = True

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
