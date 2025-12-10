from numpy import typing as npt
from lark import Token, Tree

from dataclasses import dataclass
from typing import Self, Any, Literal

from ...core.grammarclasses import CommandArguments, Command, CommandResult
from ...core.context import Context, ParseContext
from ...core.commandparser import CommandArgumentParser
from ...core.grammar.mathexpressions import Expression
from ...core.grammar.axisindexpos import AxisIndexPosition

class PlotKind:
    ...

class VariablePlotKind(PlotKind):
    x: Expression[Any]
    y: Expression[Any]
    x_variable_kind: Literal["value", "index"]

class ExpressionPlotKind(PlotKind):
    x_expression: Expression[npt.NDArray[Any]]
    y_expression: Expression[npt.NDArray[Any]]

class ResultPlotKind(PlotKind):
    result_name: str
    result_kind: str


@dataclass(slots=True)
class CommandArguments_Plot(CommandArguments):
    # What to plot
    kind: str | None

    # Where to plot (if None, create new figure not already used by another command)
    figure: AxisIndexPosition | None

    # Data manipulation (requires something to plot)
    vshift: float | None
    hshift: float | None
    voffset: float | None
    hoffset: float | None

    # Plotting color and style (requires something to plot)
    colorby: str | None
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


@dataclass(slots=True)
class CommandResult_Plot(CommandResult):
    ...

parse_plot_command = CommandArgumentParser(CommandArguments_Plot)
parse_plot_command.add_argument('kind')

parse_plot_command.add_argument('figure', '--fig', '--ax', type=AxisIndexPosition.parse, default=None)
parse_plot_command.add_argument('vshift', '--vshift', type=float, default=None)
parse_plot_command.add_argument('hshift', '--hshift', type=float, default=None)
parse_plot_command.add_argument('voffset', '--voffset', type=float, default=None)
parse_plot_command.add_argument('hoffset', '--hoffset', type=float, default=None)
parse_plot_command.add_argument('colorby', '--colorby', type=str, default=None)
parse_plot_command.add_argument('color', '--color', type=str, default=None)
parse_plot_command.add_argument('alpha', '--alpha', type=float, default=None)

parse_plot_command.add_argument('linestyle', '--linestyle', type=str, default=None)
for style in ['solid', 'dashed', 'dotted', 'dashdot']:
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
        ...

