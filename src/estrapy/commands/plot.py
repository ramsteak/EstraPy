from __future__ import annotations

import re
import numpy as np
import numpy.typing as npt
import pandas as pd

from functools import partial
from logging import getLogger
from typing import NamedTuple, Callable, Literal
from matplotlib import colormaps
from matplotlib import colors
from statsmodels.nonparametric.smoothers_lowess import lowess


from ._context import Context, Data, FigureRuntime, FigureSettings
from ._handler import CommandHandler, Token, CommandResult
from ._numberunit import NumberUnit, Bound, parse_range, NumberUnitRange
from ._misc import nderivative

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
    suptitle: str
    xlimits: NumberUnitRange | None
    ylimits: NumberUnitRange | None
    vshift: float
    colorby: str | None
    colormap: ColorMap
    alpha: float
    linewidth: float
    linestyle: str | tuple[Literal[0], tuple[int,...]]
    legend: bool
    show: bool
    grid: str|None

class ColKind(NamedTuple):
    xcol: str | None
    ycol: str
    xop:list[Callable[[npt.NDArray], npt.NDArray]]
    yop:list[Callable[[npt.NDArray, npt.NDArray], npt.NDArray]]
    xdef: str
    ydef: str

def derivative(y:npt.NDArray, x:npt.NDArray, d:int=1) -> npt.NDArray:
    return nderivative(y, d, x) # type: ignore

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
    # Check that there is an index if there are operations. If there is not, error
    if len(y_spec) > 1 and x_specs is None:
        raise ValueError("Y operations require an X column to operate on.")

    for ys in reversed(y_spec[:-1]):
        factor = int(ys[1:]) if len(ys) > 1 else 1
        match ys[0]:
            case "r": op = lambda y,_: np.real(y)
            case "i": op = lambda y,_: np.imag(y)
            case "a": op = lambda y,_: np.abs(y)
            case "p": op = lambda y,_: np.unwrap(np.angle(y))
            case "s": op = lambda y,x,f=factor: partial(smooth,       d=f)(y,x)
            case "d": op = lambda y,x,f=factor: partial(derivative,   d=f)(y,x)
            case "k": op = lambda y,x,f=factor: partial(weight,       d=f)(y,x)
            case "w": op = lambda y,x,f=factor: partial(weight_one,   d=f)(y,x)
            case "W": op = lambda y,x,f=factor: partial(weight_noavg, d=f)(y,x)
            case _: raise ValueError(f"Unknown x operation: {ys}")
        y_ops.append(op)
    ycol = y_spec[-1]

    return ColKind(xcol, ycol, x_ops, y_ops, x_specs, y_specs)

def get_column_(col:ColKind, data:Data) -> tuple[npt.NDArray, npt.NDArray]:
    domain = data._get_col_domain(col.ycol)
    x = data.get_col_(col.xcol, domain=domain)
    for op in col.xop:
        x = op(x)

    y = data.get_col_(col.ycol, domain=domain)
    for op in col.yop:
        y = op(y,x)

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
        parser.add_argument("--suptitle")
        parser.add_argument("--xlim", nargs=2)
        parser.add_argument("--ylim", nargs=2)
        parser.add_argument("--vshift", type=float, default=0)
        parser.add_argument("--colorby")
        parser.add_argument("--figure", default=None)
        parser.add_argument("--color", nargs="+")
        parser.add_argument("--alpha", type=float, default=1.0)
        parser.add_argument("--show", action="store_true")
        parser.add_argument("--legend", action="store_true")
        parser.add_argument("--gridx", action="store_true")
        parser.add_argument("--gridy", action="store_true")
        parser.add_argument("--grid", action="store_true")

        lwidth = parser.add_mutually_exclusive_group()
        lwidth.add_argument("--linewidth", type=float, default=1.0)
        lwidth.add_argument("--xxthick", action="store_const", const=8.0, dest="linewidth")
        lwidth.add_argument("--xthick", action="store_const", const=4.0, dest="linewidth")
        lwidth.add_argument("--thick", action="store_const", const=2.0, dest="linewidth")
        
        lwidth.add_argument("--thin", action="store_const", const=0.5, dest="linewidth")
        lwidth.add_argument("--xthin", action="store_const", const=0.25, dest="linewidth")
        lwidth.add_argument("--xxthin", action="store_const", const=0.125, dest="linewidth")

        lstyle = parser.add_mutually_exclusive_group()
        lstyle.add_argument("--linestyle", default="solid")
        lstyle.add_argument("--solid", action="store_const", const="solid", dest="linestyle")
        lstyle.add_argument("--dotted", action="store_const", const="dotted", dest="linestyle")
        lstyle.add_argument("--dashed", action="store_const", const="dashed", dest="linestyle")
        lstyle.add_argument("--dashdot", action="store_const", const="dashdot", dest="linestyle")
        
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
                # Double up the color to have a constant colormap
                cmap = colors.LinearSegmentedColormap.from_list("customcmap", [_color,_color])
                cmapcount = None
            case list(_colorlist):
                cmap = colors.LinearSegmentedColormap.from_list('customcmap', _colorlist)
                cmapcount = None
            case _:
                raise RuntimeError()
        colormap = ColorMap(cmap, cmapcount)
        
        # Labels
        if any(l is not None for l in (args.xlabel, args.ylabel, args.title)):
            labels = Labels(
                args.title,
                args.xlabel,
                args.ylabel,
            )
        else: labels = None

        # Figure, show
        show = args.show
        if args.figure is None:
            figsettings = FigureSettings(-1, (1,1))
            figure = figsettings
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

        # Line style
        # Try parsing as a list of numbers separated by dots
        try:
            style = (0, tuple(int(n) for n in str(args.linestyle).split(".")))
        except ValueError:
            style = args.linestyle
        
        match args.gridx, args.gridy, args.grid:
            case [False, False, False]: grid = None
            case [True, True, False] | [_, _, True]: grid = "xy"
            case [True, False, False]: grid = "x"
            case [False, True, False]: grid = "y"
            case _:raise RuntimeError("Boolean logic goes brrr")

        return Args_Plot(
            plot,
            figure,
            labels,
            args.suptitle,
            parse_range(*args.xlim) if args.xlim else None,
            parse_range(*args.ylim) if args.ylim else None,
            args.vshift,
            args.colorby,
            colormap,
            args.alpha,
            args.linewidth,
            style,
            args.legend,
            show,
            grid
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
            # Get coloring
            if args.colorby is None:
                _colorby = np.array(range(len(context.data.data))) + len(ax._lines) + 1
            else:
                _colorby = np.array([data.meta.get(args.colorby) for data in context.data])
            qualitative = np.issubdtype(_colorby.dtype, np.integer) or np.issubdtype(_colorby.dtype, np.str_)
            
            match qualitative, args.colormap.num:
                case True, None:
                # Value is qualitative and colormap is quantitative
                    uniquevals, inverse = np.unique(_colorby, return_inverse=True)
                    ncats = len(uniquevals)
                    catcolors = np.array([args.colormap.cmap(i/(max(ncats-1,1))) for i,_ in enumerate(uniquevals)])
                    colors = catcolors[inverse]
                case True, cmaplength:
                # Value is qualitative and colormap is qualitative
                    uniquevals, inverse = np.unique(_colorby, return_inverse=True)
                    ncats = len(uniquevals)
                    catcolors = np.array([args.colormap.cmap(i%cmaplength) for i,_ in enumerate(uniquevals)])
                    colors = catcolors[inverse]
                case False, _:
                # Value is quantitative
                    m,M = min(_colorby), max(_colorby)
                    if np.isclose(m-M, 0):
                        colors = args.colormap.cmap(np.zeros_like(_colorby))
                    else:
                        colors = args.colormap.cmap((_colorby-m)/(M-m))
                    pass
                    

            for i,data in enumerate(context.data):
                x,_y = get_column_(args.plot, data)
                y = _y + i * args.vshift

                ax._lines.append((x,y))
                
                ax.axis.plot(x, y, color=colors[i],
                             linewidth=args.linewidth,
                             linestyle=args.linestyle,
                             alpha=args.alpha,
                             label=data.meta.name
                            )

        if args.suptitle is not None:
            fig.suptitle(args.suptitle)

        if args.labels is not None:
            if args.labels.xlabel is not None:
                ax.axis.set_xlabel(args.labels.xlabel)
            if args.labels.ylabel is not None:
                ax.axis.set_ylabel(args.labels.ylabel)
            if args.labels.title is not None:
                ax.axis.set_title(args.labels.title)


        # Update the stored limit definitions, if not `..`
        if args.xlimits is not None:
            _al, _au = ax.xlimits
            match args.xlimits.lower:
                case NumberUnit(_l, _, _) if _l == -np.inf:...
                case _:
                    _al = args.xlimits.lower
            match args.xlimits.upper:
                case NumberUnit(_u, _, _) if _u == np.inf:...
                case _:
                    _au = args.xlimits.upper
            ax.xlimits = (_al, _au)
            
        if args.ylimits is not None:
            _al, _au = ax.ylimits
            match args.ylimits.lower:
                case NumberUnit(_l, _, _) if _l == -np.inf:...
                case _:
                    _al = args.ylimits.lower
            match args.ylimits.upper:
                case NumberUnit(_u, _, _) if _u == np.inf:...
                case _:
                    _au = args.ylimits.upper
            ax.ylimits = (_al, _au)
        
        # Act on the limit definitions, if the limit is not None or inf / -inf
        match ax.xlimits[0]:
            case Bound.EXTERNAL:
                _xlow = min([np.min(x) for x,_ in ax._lines])
            case Bound.INTERNAL:
                _xlow = max([np.min(x) for x,_ in ax._lines])
            case NumberUnit(_l,_,_) if _l != -np.inf:
                _xlow = _l
            case _:
                _xlow = None
        if _xlow is not None:
            ax.axis.set_xlim(left = _xlow)

        match ax.xlimits[1]:
            case Bound.EXTERNAL:
                _xhig = max([np.max(x) for x,_ in ax._lines])
            case Bound.INTERNAL:
                _xhig = min([np.max(x) for x,_ in ax._lines])
            case NumberUnit(_l,_,_) if _l != np.inf:
                _xhig = _l
            case _:
                _xhig = None
        if _xhig is not None:
            ax.axis.set_xlim(right = _xhig)

        _xlow,_xhig = ax.axis.get_xlim()
        
        match ax.ylimits[0]:
            case Bound.EXTERNAL:
                _ylow = min([np.min(y[(x>=_xlow)&(x<=_xhig)]) for x,y in ax._lines])
            case Bound.INTERNAL:
                _ylow = max([np.min(y[(x>=_xlow)&(x<=_xhig)]) for x,y in ax._lines])
            case NumberUnit(_l,_,_) if _l != -np.inf:
                _ylow = _l
            case _:
                _ylow = None

        match ax.ylimits[1]:
            case Bound.EXTERNAL:
                _yhig = max([np.max(y[(x>=_xlow)&(x<=_xhig)]) for x,y in ax._lines])
            case Bound.INTERNAL:
                _yhig = min([np.max(y[(x>=_xlow)&(x<=_xhig)]) for x,y in ax._lines])
            case NumberUnit(_l,_,_) if _l != np.inf:
                _yhig = _l
            case _:
                _yhig = None
        
        match ax.ylimits[0], ax.ylimits[1]:
            case Bound(), Bound():
                # Expand both limits by 5%
                _d = _yhig-_ylow # type: ignore
                _yhig = _yhig + _d * 0.05 # type: ignore
                _ylow = _ylow - _d * 0.05 # type: ignore
            case Bound(), _:
                # Expand lower limit by 5% relative to current upper ylim
                _,_u = ax.axis.get_ylim()
                _d = _u-_ylow # type: ignore
                _ylow = _ylow - _d * 0.05 # type: ignore
            case _, Bound():
                # Expand upper limit by 5% relative to current lower ylim
                _l,_ = ax.axis.get_ylim()
                _d = _yhig-_l # type: ignore
                _yhig = _yhig + _d * 0.05 # type: ignore

        if _ylow is not None:
            ax.axis.set_ylim(bottom = _ylow)
        if _yhig is not None:
            ax.axis.set_ylim(top = _yhig)
        
        if args.legend:
            ax.axis.legend()
        
        if args.grid is not None:
            if "x" in args.grid:ax.axis.grid(True, axis="x")
            if "y" in args.grid:ax.axis.grid(True, axis="y")
        

        if args.show:
            figure.show()

        return CommandResult(True)

    @staticmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult:
        # TODO:
        raise NotImplementedError
