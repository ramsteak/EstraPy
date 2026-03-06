import numpy as np
import pandas as pd

from numpy import typing as npt

from lark import Token, Tree

from dataclasses import dataclass
from typing import Self, Any

from ...core.context import Command, CommandResult
from ...core.context import Context, ParseContext, PlotContext, FigureSpecification, AxisSpecification
from ...core.datastore import DataPage
from ...core.commandparser import CommandArgumentParser, CommandArguments, field_arg
from ...core._validators import validate_float_positive, validate_float_non_negative, type_bool
from ...core.grammar.mathexpressions import Expression
from ...core.grammar.axisindexpos import AxisIndexPosition
from ...core.misc import template_replace
from ...core.number import Number


def parse_x_tuple_floats(value: str) -> tuple[float, float]:
    a,b = value.split('x')
    return (float(a), float(b))

@dataclass(slots=True)
class CommandArguments_Plot(CommandArguments):
    kind: str | None = field_arg(
        type=str,
        required=False,
        default=None
    )

    subkind: str | None = field_arg(
        type=str,
        required=False,
        default=None
    )

    figure: AxisIndexPosition | None = field_arg(
        flags=['--figure', '--fig', '--ax'],
        type=AxisIndexPosition.parse,
        required=False,
        default=None
    )

    vshift: float | None = field_arg(
        flags=['--vshift'],
        type=float,
        required=False,
        default=None
    )

    hshift: float | None = field_arg(
        flags=['--hshift'],
        type=float,
        required=False,
        default=None
    )

    voffset: float | None = field_arg(
        flags=['--voffset'],
        type=float,
        required=False,
        default=None
    )

    hoffset: float | None = field_arg(
        flags=['--hoffset'],
        type=float,
        required=False,
        default=None
    )

    colorby: str | None = field_arg(
        flags=['--colorby'],
        type=str,
        required=False,
        default=None
    )

    legendname: str | None = field_arg(
        flags=['--legendname'],
        type=str,
        required=False,
        default=None
    )

    color: str | None = field_arg(
        flags=['--color'],
        type=str,
        required=False,
        default=None
    )

    alpha: float | None = field_arg(
        flags=['--alpha'],
        type=float,
        required=False,
        default=None,
        validate=validate_float_non_negative # Alpha typically 0.0 to 1.0
    )

    linestyle: str | None = field_arg(
        flags=['--linestyle'],
        type=str,
        required=False,
        default=None,
        const_flags={
            '--solid': 'solid', 
            '--dashed': 'dashed', 
            '--dotted': 'dotted', 
            '--dashdot': 'dashdot', 
            '--noline': 'noline'
        }
    )

    linewidth: float | None = field_arg(
        flags=['--linewidth'],
        type=float,
        required=False,
        default=None,
        const_flags={
            '--xxthin': 0.125,
            '--xthin': 0.25,
            '--thin': 0.5,
            '--medium': 1.0,
            '--thick': 2.0,
            '--xthick': 4.0,
            '--xxthick': 8.0
        },
        validate=validate_float_positive
    )

    marker: str | None = field_arg(
        flags=['--marker'],
        type=str,
        required=False,
        default=None,
        const_flags={
            '--none': 'none',
            '--nomarker': 'none',
            '--circle': 'o',
            '--square': 's', 
            '--triangle_up': '^',
            '--triangle_down': 'v', 
            '--diamond': 'D'
        }
    )

    markersize: float | None = field_arg(
        flags=['--markersize'],
        type=float,
        required=False,
        default=None,
        validate=validate_float_positive
    )

    markeredgecolor: str | None = field_arg(
        flags=['--markeredgecolor'],
        type=str,
        required=False,
        default=None
    )

    markerfacecolor: str | None = field_arg(
        flags=['--markerfacecolor'],
        type=str,
        required=False,
        default=None
    )

    xlabel: str | None = field_arg(
        flags=['--xlabel'],
        type=str,
        required=False,
        default=None
    )

    ylabel: str | None = field_arg(
        flags=['--ylabel'],
        type=str,
        required=False,
        default=None
    )

    title: str | None = field_arg(
        flags=['--title'],
        type=str,
        required=False,
        default=None
    )

    suptitle: str | None = field_arg(
        flags=['--suptitle'],
        type=str,
        required=False,
        default=None
    )

    legend: bool | None = field_arg(
        flags=['--legend'],
        type=type_bool,
        const=True,
        required=False,
        default=None,
        nargs='?'
    )

    xlim: tuple[float, float] | None = field_arg(
        flags=['--xlim'],
        type=float,
        nargs=2,
        required=False,
        default=None
    )

    ylim: tuple[float, float] | None = field_arg(
        flags=['--ylim'],
        type=float,
        nargs=2,
        required=False,
        default=None
    )

    grid: bool | None = field_arg(
        flags=['--grid'],
        type=type_bool,
        const=True,
        required=False,
        default=None,
        nargs='?'
    )

    xgrid: bool | None = field_arg(
        flags=['--xgrid'],
        type=type_bool,
        const=True,
        required=False,
        default=None,
        nargs='?'
    )

    ygrid: bool | None = field_arg(
        flags=['--ygrid'],
        type=type_bool,
        const=True,
        required=False,
        default=None,
        nargs='?'
    )

    figsize: tuple[float, float] | None = field_arg(
        flags=['--figsize'],
        type=parse_x_tuple_floats,
        required=False,
        default=None
    )

    def validate(self) -> None:
        if self.kind is None:
            if any([
                self.vshift is not None,
                self.hshift is not None,
                self.voffset is not None,
                self.hoffset is not None,
                self.colorby is not None,
                self.color is not None,
                self.alpha is not None,
                self.linestyle is not None,
                self.linewidth is not None,
                self.marker is not None,
                self.markersize is not None,
                self.markeredgecolor is not None,
                self.markerfacecolor is not None,
            ]):
                raise ValueError("Plot command: If no 'kind' is specified, no data manipulation or styling options can be used.")



@dataclass(slots=True)
class CommandResult_Plot(CommandResult):
    ...

parse_plot_command = CommandArgumentParser(CommandArguments_Plot, 'plot')


def freeze_to_vars(
        page: DataPage,
        *,
        include_columns: bool = True,
        include_hist_columns: bool = True,
        include_vars: bool = True
    ) -> dict[str, Any]:

    vars: dict[str, Any] = {}
    if include_vars:
        vars |= {k.replace(".","_"):v for k,v in page.meta._dict.items()} # pyright: ignore[reportPrivateUsage]
    
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

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from matplotlib.colors import Colormap

def parse_colormap(cmap: str) -> "Colormap":
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.cm import get_cmap

    if ',' in cmap and '..' in cmap:
        raise ValueError("Plot command: 'color' argument cannot contain both ',' and '..' separators.")
    elif '..' in cmap:
        # LinearSegmentedColormap
        colors = [c.strip() for c in cmap.split('..')]
        return LinearSegmentedColormap.from_list(f'lscm_{"_".join(colors)}', colors)
    elif ',' in cmap:
        # ListedColormap
        colors = [c.strip() for c in cmap.split(',')]
        return ListedColormap(colors, name=f'lcm_{"_".join(colors)}')
    else:
        # Single color or colormap name
        colorstr = cmap.strip()
        try:
            return get_cmap(colorstr)
        except ValueError:
            try:
                # Test if it is a valid color by trying to create a colormap with it
                return LinearSegmentedColormap.from_list(f'{colorstr}', [colorstr, colorstr])
            except ValueError:
                raise ValueError(f"Plot command: 'color' argument '{cmap}' is not a valid colormap name, color specification, or list of colors.") from None

@dataclass(slots=True)
class Command_Plot(Command[CommandArguments_Plot, CommandResult_Plot]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_plot_command.parse(commandtoken, tokens)

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
            return self._execute_plot_result(context, ifigure, iaxis)
        else:
            return self._execute_plot_expression(context, ifigure, iaxis)


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
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure
        from matplotlib.colors import LinearSegmentedColormap, ListedColormap
        from matplotlib import colors
        from matplotlib.cm import get_cmap

        import seaborn as sns


        # Apply axis test annotations if specified
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
        elif self.args.xgrid is False:
            iaxis.callbacks.append(lambda ax, fig: ax.grid(axis='x', which='both', visible=False)) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        if self.args.ygrid is True:
            iaxis.callbacks.append(lambda ax, fig: ax.grid(axis='y', which='both', visible=True)) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        elif self.args.ygrid is False:
            iaxis.callbacks.append(lambda ax, fig: ax.grid(axis='y', which='both', visible=False)) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        if self.args.grid is True:
            iaxis.callbacks.append(lambda ax, fig: ax.grid(which='both', visible=True)) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        elif self.args.grid is False:
            iaxis.callbacks.append(lambda ax, fig: ax.grid(which='both', visible=False)) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        
        # Limits specifications
        if self.args.xlim is not None:
            iaxis.callbacks.append(lambda ax, fig: ax.set_xlim(self.args.xlim)) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        if self.args.ylim is not None:
            iaxis.callbacks.append(lambda ax, fig: ax.set_ylim(self.args.ylim)) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        

        # Determine the color variable
        if self.args.colorby is not None:
            if '{' in self.args.colorby:
                colorby_var = [template_replace(self.args.colorby, page.meta) for page in context.datastore.pages.values()]
            else:
                exp = Expression[Any].compile(self.args.colorby)
                colorby_var = [exp(**freeze_to_vars(page, include_columns=False, include_hist_columns=False)) for page in context.datastore.pages.values()]
        else:
            colorby_var = list(range(len(context.datastore.pages)))

        colorby_var = [float(v) if isinstance(v, Number) else v for v in colorby_var]

        # Determine the color palette
        color_categorical = any(isinstance(v, (str, int)) for v in colorby_var)
        if self.args.color is not None:
            colormap = parse_colormap(self.args.color)
        else:
            colormap = get_cmap('tab10') if color_categorical else get_cmap('viridis')

        # Determine the color values for each page based on the colorby variable and the colormap
        if color_categorical:
            unique_cats, unique_cat_index = np.unique(colorby_var, return_inverse=True)

            # For categorical variables, we use the colormap as a list of colors and map each unique value to a color
            if isinstance(colormap, ListedColormap) and len(colormap.colors) <= 25: # Arbitrary limit for "enough colors in the colormap" to become a sequential mapping instead of a categorical mapping
                palette = {unique_cats[i]: colormap.colors[i % len(colormap.colors)] for i in range(len(unique_cats))}
            else:
                _norm = unique_cats.shape[0]-1 if unique_cats.shape[0] > 1 else 1
                palette = {unique_cats[i]: colormap(i / _norm) for i in range(len(unique_cats))}
        else:
            # For continuous variables, we normalize the values to the range 0,1 and give a sampleable colormap that supports getitem
            norm = colors.Normalize(min(colorby_var), max(colorby_var))

            class SampleableColormap:
                def __init__(self, cmap: colors.Colormap, norm: colors.Normalize):
                    self.cmap = cmap
                    self.norm = norm
                
                def __getitem__(self, value: float) -> Any:
                    return colors.to_hex(self.cmap(self.norm(value)))
            palette = SampleableColormap(colormap, norm)

            
        # Plotting logic

        # Freeze the pages into variable dictionaries for expression evaluation (both kind and color)
        frozenpages = {name: freeze_to_vars(page) for name, page in context.datastore.pages.items()}

        # An expression requires xaxis:yaxis, separated by a colon, so we can assume that if there
        # is a colon and both parts parse as expressions, it is an expression plot (either variable or spectra).
        if isinstance(self.args.kind, str) and ':' in self.args.kind:
            xpart, ypart = self.args.kind.split(':', 1)
            # Try to parse both parts as expressions
            xexpr = Expression[float | npt.NDArray[np.floating]].compile(xpart)
            yexpr = Expression[float | npt.NDArray[np.floating]].compile(ypart)


            pass

            datas: list[pd.DataFrame] = [
                pd.DataFrame(
                        {
                            'unit': name,
                            'x': np.atleast_1d(xexpr(**fpage)),
                            'y': np.atleast_1d(yexpr(**fpage) + (self.args.voffset or 0) + (self.args.vshift or 0) * i),
                            'color': color
                        }
                    )
                for i, ((name, fpage), color) in enumerate(zip(frozenpages.items(), colorby_var))
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
                sns.lineplot(data_flat,
                             x='x', y='y',
                             hue='color',
                             units='unit' if multiple_lines else None,
                             palette=palette,
                             estimator=None,
                             ax=ax, **style)
                if self.args.legend is True:
                    ax.legend(title=self.args.legendname)
                elif self.args.legend is False:
                    ax.get_legend().remove()
            
            iaxis.callbacks.append(cb)
        