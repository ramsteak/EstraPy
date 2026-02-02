import numpy as np
import pandas as pd

from numpy import typing as npt
from lark import Token, Tree

from dataclasses import dataclass
from typing import Self, Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.cm import get_cmap

import seaborn as sns

from ...core.context import CommandArguments, Command, CommandResult
from ...core.context import Context, ParseContext, PlotContext, FigureSpecification, AxisSpecification
from ...core.datastore import DataPage
from ...core.commandparser import CommandArgumentParser
from ...core.grammar.mathexpressions import Expression
from ...core.grammar.axisindexpos import AxisIndexPosition
from ...core.misc import template_replace
from ...core.number import Number

@dataclass(slots=True)
class CommandArguments_Plot(CommandArguments):
    # What to plot
    kind: str | None
    subkind: str | None

    # Where to plot (if None, create new figure not already used by another command)
    figure: AxisIndexPosition | None

    # Data manipulation (requires something to plot)
    vshift: float | None
    hshift: float | None
    voffset: float | None
    hoffset: float | None

    # Plotting color and style (requires something to plot)
    colorby: str | None
    legendname: str | None
    color: str | None
    alpha: float | None
    linestyle: str | None
    linewidth: float | None
    marker: str | None
    markersize: float | None
    markeredgecolor: str | None
    markerfacecolor: str | None

    # Plot text customization (only requires a figure/axis)
    xlabel: str | None      
    ylabel: str | None      
    title: str | None       
    suptitle: str | None    
    legend: bool | None      

    # Axes customization (only requires a figure/axis)
    xlim: tuple[float, float] | None
    ylim: tuple[float, float] | None
    xgrid: bool | None
    ygrid: bool | None
    grid: bool | None

    # Figure customization
    figsize: tuple[float, float] | None


@dataclass(slots=True)
class CommandResult_Plot(CommandResult):
    ...

parse_plot_command = CommandArgumentParser(CommandArguments_Plot)
parse_plot_command.add_argument('kind')
parse_plot_command.add_argument('subkind', default=None, required=False)

parse_plot_command.add_argument('figure', '--fig', '--ax', type=AxisIndexPosition.parse, default=None)
parse_plot_command.add_argument('vshift', '--vshift', type=float, default=None)
parse_plot_command.add_argument('hshift', '--hshift', type=float, default=None)
parse_plot_command.add_argument('voffset', '--voffset', type=float, default=None)
parse_plot_command.add_argument('hoffset', '--hoffset', type=float, default=None)
parse_plot_command.add_argument('colorby', '--colorby', type=str, default=None)
parse_plot_command.add_argument('legendname', '--legendname', type=str, default=None)
parse_plot_command.add_argument('color', '--color', type=str, default=None)
parse_plot_command.add_argument('alpha', '--alpha', type=float, default=None)

parse_plot_command.add_argument('linestyle', '--linestyle', type=str, default=None)
for style in ['solid', 'dashed', 'dotted', 'dashdot', 'noline']:
    parse_plot_command.add_argument(None, f'--{style}', type=str, action='store_const', dest='linestyle', const=style, nargs=0)

parse_plot_command.add_argument('linewidth', '--linewidth', type=float, default=None)
for widthname,width in [('xxthin', 0.125),('xthin', 0.25),('thin', 0.5),('medium', 1.0),('thick', 2.0),('xthick', 4.0),('xxthick', 8.0)]:
    parse_plot_command.add_argument(None, f'--{widthname}', type=str, action='store_const', dest='linewidth', const=width, nargs=0)

parse_plot_command.add_argument('marker', '--marker', type=str, default=None)
for markname,marker in [('none', 'none'), ('nomarker', 'none'), ('circle', 'o'), ('square', 's'), ('triangle_up', '^'), ('triangle_down', 'v'), ('diamond', 'D')]:
    parse_plot_command.add_argument(None, f'--{markname}', type=str, action='store_const', dest='marker', const=marker, nargs=0)

parse_plot_command.add_argument('markersize', '--markersize', type=float, default=None)
parse_plot_command.add_argument('markeredgecolor', '--markeredgecolor', type=str, default=None)
parse_plot_command.add_argument('markerfacecolor', '--markerfacecolor', type=str, default=None)

parse_plot_command.add_argument('xlabel', '--xlabel', type=str, default=None)
parse_plot_command.add_argument('ylabel', '--ylabel', type=str, default=None)
parse_plot_command.add_argument('title', '--title', type=str, default=None)
parse_plot_command.add_argument('suptitle', '--suptitle', type=str, default=None)
parse_plot_command.add_argument('legend', '--legend', action='set_true', default=None, nargs=0)

parse_plot_command.add_argument('xlim', '--xlim', nargs=2, type=float, default=None)
parse_plot_command.add_argument('ylim', '--ylim', nargs=2, type=float, default=None)
parse_plot_command.add_argument('grid', '--grid', action='set_true', default=None, nargs=0)
parse_plot_command.add_argument('xgrid', '--xgrid', action='set_true', default=None, nargs=0)
parse_plot_command.add_argument('ygrid', '--ygrid', action='set_true', default=None, nargs=0)

def parse_x_tuple_floats(value: str) -> tuple[float, float]:
    a,b = value.split('x')
    return (float(a), float(b))
parse_plot_command.add_argument('figsize', '--figsize', type=parse_x_tuple_floats, default=None)

def freeze_to_vars(
        page: DataPage,
        *,
        include_columns: bool = True,
        include_hist_columns: bool = True,
        include_vars: bool = True
    ) -> dict[str, Any]:

    vars: dict[str, Any] = {}
    if include_vars:
        vars |= {k.replace(".","_"):v for k,v in page.meta._dict.items()}
    
    if include_columns or include_hist_columns:
        for domain in page.domains.values():
            # Freeze raw names of the data columns
            if include_hist_columns:
                # The columns in the dataframe are the whole history of the data
                vars |= {str(n):s.values for n,s in domain.data.items()} # pyright: ignore[reportArgumentType]

            # Freeze current names for the columns. The last entry in the history is the current one
            if include_columns:
                vars.update({vname: vars.get(collist[-1].meta.physicalname) for vname, collist in domain.columns.items()})

    return vars # pyright: ignore[reportPrivateUsage]

@dataclass(slots=True)
class Command_Plot(Command[CommandArguments_Plot, CommandResult_Plot]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_plot_command(commandtoken, tokens, parsecontext)

        # If no kind is specified, no options that require data to plot should be used
        if arguments.kind is None:
            if any([
                arguments.vshift is not None,
                arguments.hshift is not None,
                arguments.voffset is not None,
                arguments.hoffset is not None,
                arguments.colorby is not None,
                arguments.color is not None,
                arguments.alpha is not None,
                arguments.linestyle is not None,
                arguments.linewidth is not None,
                arguments.marker is not None,
                arguments.markersize is not None,
                arguments.markeredgecolor is not None,
                arguments.markerfacecolor is not None,
            ]):
                raise ValueError("Plot command: If no 'kind' is specified, no data manipulation or styling options can be used.")

        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Plot:
        # Shorthand
        plotcontext: PlotContext = context.plotcontext

        # First, we determine the figure and axis to plot on
        if self.args.figure is not None:
            fignum = self.args.figure.figurenumber
            # Use the given fignum. Create it if it does not exist.
            ifigure = plotcontext.numberedfigures.setdefault(fignum, FigureSpecification())
            # Check if the axis exists in the figure
            iaxis = ifigure.axes.setdefault(self.args.figure.axisindex, AxisSpecification(pos=self.args.figure))
        else:
            # Since a figure number was not specified, create a new figure as a non-numbered figure
            ifigure = FigureSpecification()
            plotcontext.nonnumberedfigures.append(ifigure)

            iaxis = AxisSpecification(pos=AxisIndexPosition(-1, (1,1), (None, None)))
            ifigure.axes[(1,1)] = iaxis

        
        # Apply figure size if specified
        if self.args.figsize is not None:
            ifigure.figsize = self.args.figsize
        

        if self.args.kind == 'result':
            self._execute_plot_result(context, ifigure, iaxis)
        else:
            self._execute_plot_expression(context, ifigure, iaxis)

    def _execute_plot_result(self, context: Context, ifig: FigureSpecification, iaxis: AxisSpecification) -> CommandResult_Plot:
        if self.args.subkind is None:
            raise ValueError("Plot command: When plotting a 'result', a result name must be specified as 'kind:subkind' or 'kind'.")
        
        resultname, *resultkind = self.args.subkind.split(".", maxsplit=1)
        resultkind = resultkind[0] if resultkind else None

        # Check if the result exists
        if resultname not in context.results:
            raise ValueError(f"Plot command: No result named '{self.args.subkind}' found.")
        
        result = context.results[resultname]

        # Check if the subkind is valid for the result.
        # Results implement callback-factories for plotting.

        # Introspect the result to list all subkind plots

        valid_reskinds = {
            e.removeprefix("plot").removeprefix("_") or None: getattr(result, e)
            for e in dir(result)
            if callable(getattr(result, e)) and e.startswith("plot")
        }

        if resultkind not in valid_reskinds:
            raise ValueError(f"Plot command: Result '{resultname}' has no plot of kind '{resultkind}'. Valid kinds are: {', '.join(repr(k) for k in valid_reskinds.keys())}.")
        
        plot_callback_factory = valid_reskinds[resultkind]
        iaxis.callbacks.append(plot_callback_factory())

        return CommandResult_Plot()

    def _execute_plot_expression(self, context: Context, ifig: FigureSpecification, iaxis: AxisSpecification) -> CommandResult_Plot:
        # Apply axis settings if specified
        if self.args.xlabel is not None:
            iaxis.callbacks.append(lambda ax, fig: ax.set_xlabel(self.args.xlabel)) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        if self.args.ylabel is not None:
            iaxis.callbacks.append(lambda ax, fig: ax.set_ylabel(self.args.ylabel)) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        if self.args.title is not None:
            iaxis.callbacks.append(lambda ax, fig: ax.set_title(self.args.title)) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        if self.args.suptitle is not None:
            iaxis.callbacks.append(lambda ax, fig: fig.suptitle(self.args.suptitle)) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        
        # Grid specifications
        if self.args.xgrid is True:
            iaxis.callbacks.append(lambda ax, fig: ax.grid(axis='x', which='both', visible=True)) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        if self.args.ygrid is True:
            iaxis.callbacks.append(lambda ax, fig: ax.grid(axis='y', which='both', visible=True)) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        if self.args.grid is True:
            iaxis.callbacks.append(lambda ax, fig: ax.grid(which='both', visible=True)) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        
        # Limits specifications
        if self.args.xlim is not None:
            iaxis.callbacks.append(lambda ax, fig: ax.set_xlim(self.args.xlim)) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        if self.args.ylim is not None:
            iaxis.callbacks.append(lambda ax, fig: ax.set_ylim(self.args.ylim)) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        
        # Color logic
        # Determine the colorby variable values
        # Colorby can be either a single variable name, a composite file name or an expression.
        # We need to differentiate and get the single values for each data page.

        # We assume that if "{}" is in the string, it is a composite filename.
        # If not, we try to parse it as an expression. A single variable should parse as an expression too.
        if self.args.colorby is not None:
            if '{' in self.args.colorby:
                # Composite filename
                colorby_var = [
                    template_replace(self.args.colorby, page.meta)
                    for page in context.datastore.pages.values()
                ]
            else:
                exp = Expression[Any].compile(self.args.colorby)
                colorby_var = [
                    exp(**freeze_to_vars(page, include_columns=False, include_hist_columns=False))
                    for page in context.datastore.pages.values()
                ]
        else:
            # Colorby is not specified, use the index of the data pages
            colorby_var = list(range(len(context.datastore.pages)))

        # Convert Number instances to float
        colorby_var = [float(v) if isinstance(v, Number) else v for v in colorby_var]

        # Determine wether the colorby variable is categorical or continuous
        is_categorical = any(isinstance(v, (str, int)) for v in colorby_var)
        
        pass


        if self.args.color is not None:
            # Color can either be:
            #  - a sequence of colors (either named or #rrggbb) separated by , -> ListedColormap
            #  - a sequence of colors (either named or #rrggbb) separated by .. -> LinearSegmentedColormap
            #  - a named color (or #rrggbb) -> LinearSegmentedColormap with two identical colors
            #  - a matplotlib colormap name -> colormap
            #  - a matplotlib reversed colormap name (e.g. 'viridis_r') -> colormap
            
            # self.args.color is a single string with either "," or ".." or neither.
            if ',' in self.args.color and '..' in self.args.color:
                raise ValueError("Plot command: 'color' argument cannot contain both ',' and '..' separators.")
            elif '..' in self.args.color:
                # LinearSegmentedColormap
                colors = [c.strip() for c in self.args.color.split('..')]
                colormap = LinearSegmentedColormap.from_list(f'lscm_{"_".join(colors)}', colors)
            elif ',' in self.args.color:
                # ListedColormap
                colors = [c.strip() for c in self.args.color.split(',')]
                colormap = ListedColormap(colors, name=f'lcm_{"_".join(colors)}')
            else:
                # Single color or colormap name
                colorstr = self.args.color.strip()
                try:
                    colormap = get_cmap(colorstr)
                    
                except ValueError:
                    # Not a colormap name, use two identical colors
                    colormap = LinearSegmentedColormap.from_list(f'{colorstr}', [colorstr, colorstr])
        else:
            # Default colormap
            colormap = get_cmap('tab10') if is_categorical else get_cmap('viridis')

        
        # Plotting logic
        # Freeze the pages into variable dictionaries for expression evaluation (both kind and color)
        frozenpages = {name: freeze_to_vars(page) for name, page in context.datastore.pages.items()}
        # An expression requires xaxis:yaxis, separated by a colon, so we can assume that if there
        # is a colon and both parts parse as expressions, it is an expression plot (either variable or spectra).
        if isinstance(self.args.kind, str) and ':' in self.args.kind:
            xpart, ypart = self.args.kind.split(':', 1)
            # Try to parse both parts as expressions
            xexpr = Expression.compile(xpart)
            yexpr = Expression.compile(ypart)

            # xaxis: list[npt.NDArray[np.floating] | float] = []
            # yaxis: list[npt.NDArray[np.floating] | float] = []

            # Collect dataframes for each page, with columns:
            #  - 'unit' -> the name of the page, to be used with sns.unit
            #  - 'x', 'y' -> the x and y values
            #  - 'color' -> the colorby variable value
            # The dataframe is then concatenated and plotted with seaborn

            pass

            datas: list[pd.DataFrame] = [
                pd.DataFrame(
                        {
                            'unit': name,
                            'x': np.atleast_1d(xexpr(**fpage)),
                            'y': np.atleast_1d(yexpr(**fpage)),
                            'color': color
                        }
                    )
                for (name, fpage), color in zip(frozenpages.items(), colorby_var)
            ]

            style: dict[str, Any] = {}
            if self.args.linestyle is not None:
                if self.args.linestyle == 'noline':
                    style['linestyle'] = 'none'
                else:
                    style['linestyle'] = self.args.linestyle
            
            if self.args.linewidth is not None:
                style['linewidth'] = self.args.linewidth
            
            if self.args.marker is not None:
                style['marker'] = self.args.marker

            # Determine if the plot is
            #  - a collection of line plots -> each dataframe is more than one point
            #  - one plot -> each dataframe is a single point
            multiple_lines = any(len(df) > 1 for df in datas)

            data_flat = pd.concat(datas, ignore_index=True)

            def cb(ax: Axes, fig: Figure) -> Any:
                if is_categorical:
                    palette = colormap.colors if isinstance(colormap, ListedColormap) else None
                else:
                    palette = colormap
                sns.lineplot(data_flat,
                             x='x', y='y',
                             hue='color',
                             units='unit' if multiple_lines else None,
                             palette=palette,
                             ax=ax)
            
            iaxis.callbacks.append(cb)
        