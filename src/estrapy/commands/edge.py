import numpy as np

from numpy import typing as npt
from logging import Logger
from dataclasses import dataclass
from scipy.interpolate import make_interp_spline, BSpline # type: ignore
from lark import Token, Tree
from typing import Self
from functools import partial

from ..core.datastore import Domain, ColumnDescription, ColumnKind, DataPage
from ..core.context import CommandArguments, Command, CommandResult
from ..core.context import Context, ParseContext
from ..core.number import Number, parse_number, Unit, parse_range, parse_edge
from ..core.threaded import execute_threaded
from ..core.commandparser import CommandArgumentParser
from ..operations.edge_detection import correlation_edge_detection, SlidingL2Result
from ..operations.axis_conversions import E_to_k


@dataclass(slots=True)
class SubCommandArguments_Edge_Calc(CommandArguments):
    mode: str
    energy: Number | str
    search: Number
    delta: Number


@dataclass(slots=True)
class SubCommandArguments_Edge_Shift(CommandArguments):
    range: tuple[Number, Number]
    resolution: Number
    shift: Number
    derivative: int


@dataclass(slots=True)
class SubCommandArguments_Edge_Set(CommandArguments):
    energy: Number


@dataclass(slots=True)
class CommandArguments_Edge(CommandArguments):
    mode: SubCommandArguments_Edge_Calc | SubCommandArguments_Edge_Shift | SubCommandArguments_Edge_Set


sub_set = CommandArgumentParser(SubCommandArguments_Edge_Set, name='set')
sub_set.add_argument('energy', type=parse_edge, required=True)

sub_calc = CommandArgumentParser(SubCommandArguments_Edge_Calc, name='calc')
sub_calc.add_argument('mode', '--mode', '-m', type=str, required=False, default='set')
sub_calc.add_argument('energy', '--energy', '--E0', '-E', type=parse_edge, required=False)
sub_calc.add_argument('search', '--search', '--sE0', type=parse_edge, required=False, default=Number(None, 0.0, None))
sub_calc.add_argument('delta', '--delta', '-d', type=parse_number, required=False)

_default_range = (Number(None, -np.inf, None), Number(None, np.inf, None))
sub_shift = CommandArgumentParser(SubCommandArguments_Edge_Shift, name='shift')
sub_shift.add_argument('range', types=parse_range, nargs=2, required=False, default=_default_range)
sub_shift.add_argument('resolution', '--resolution', '--res', type=parse_number, required=False, default=Number(None, 0.1, None))
sub_shift.add_argument('shift', '--shift', '-s', type=parse_number, required=False, default=Number(None, 5.0, None))
sub_shift.add_argument('derivative', '--derivative', '--deriv', type=int, required=False, default=1)

parse_edge_command = CommandArgumentParser(CommandArguments_Edge, name='edge')
parse_edge_command.add_subparser('set', sub_set, 'mode')
parse_edge_command.add_subparser('calc', sub_calc, 'mode')
parse_edge_command.add_subparser('shift', sub_shift, 'mode')

@dataclass(slots=True)
class SubCommandResult_Edge_Shift(CommandResult):
    e_range: tuple[float, float]
    results: dict[str, SlidingL2Result]

@dataclass(slots=True)
class SubCommandResult_Edge_Set(CommandResult):
    pass

CommandResult_Edge = SubCommandResult_Edge_Shift | SubCommandResult_Edge_Set

def _get_shift_axis(range: tuple[float, float], resolution: float) -> npt.NDArray[np.floating]:
    return np.arange(range[0], range[1] + resolution, resolution)

def _set_edge_energy_value(page: DataPage, edge_energy: float) -> None:
    page.meta['E0'] = edge_energy
    domain = page.domains[Domain.RECIPROCAL]
    e_column = ColumnDescription('e', Unit.EV, ColumnKind.AXIS, deps=['E'], calc=lambda df, E0=edge_energy: df['E'] - E0, labl='Relative energy [eV]', relative=True)
    k_column = ColumnDescription('k', Unit.A, ColumnKind.AXIS, deps=['E'], calc=lambda df, E0=edge_energy: E_to_k(df['E'], E0), labl='Wave vector k [1/Å]')
    domain.add_column('e', e_column)
    domain.add_column('k', k_column)

def _get_interpolator_splines_for_pages(pages: dict[str, DataPage], range: tuple[float, float], dom: Domain, xcol:str, ycol1:str, ycol2:str, log: Logger) -> dict[str, tuple[BSpline, BSpline]]:
    splines: dict[str, tuple[BSpline, BSpline]] = {}
    for name, page in pages.items():
        domain = page.domains[dom]
        df = domain.get_columns_data([xcol, ycol1, ycol2])
        index = (df[xcol] >= range[0]) & (df[xcol] <= range[1])

        region = df[index]
        if not region.size:
            log.warning(f'No data in the specified range for page {name}. It will be skipped, this may lead to further errors. Consider adjusting the range.')
            continue

        spline1:BSpline = make_interp_spline(region[xcol], region[ycol1], k=3) # type: ignore
        spline1.extrapolate = True

        spline2:BSpline = make_interp_spline(region[xcol], region[ycol2], k=3) # type: ignore
        spline2.extrapolate = True

        splines[name] = (spline1, spline2)

    return splines

def _calculate_l2_shift_from_data(data: npt.NDArray[np.floating], reference: npt.NDArray[np.floating], name: str, derivative: int, slide_amount: int, resolution: float, log: Logger) -> SlidingL2Result:
    result = correlation_edge_detection(data, reference, derivative, slide_amount, resolution)
    log.debug(f'Calculated shift for page {name}: {result.x:0.4f} eV')
    return result

@dataclass(slots=True)
class Command_Edge(Command[CommandArguments_Edge, CommandResult_Edge]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_edge_command(commandtoken, tokens, parsecontext)
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Edge:
        match self.args.mode:
            case SubCommandArguments_Edge_Set():
                return self._execute_set(self.args.mode, context)
            case SubCommandArguments_Edge_Calc():
                raise NotImplementedError("Edge calc command is not yet implemented.")
            case SubCommandArguments_Edge_Shift():
                return self._execute_shift(self.args.mode, context)
            case _:
                raise NotImplementedError(f"Unknown mode {self.args.mode} in edge command.")
    
    def _execute_set(self, args: SubCommandArguments_Edge_Set, context: Context) -> SubCommandResult_Edge_Set:
        log = context.logger.getChild('command.edge.set')

        energy = args.energy
        if not (energy.unit is None or energy.unit == Unit.EV):
            raise ValueError("Only eV unit is supported for E0 setting.")
        if energy.sign is not None:
            raise ValueError("Relative E0 setting is not supported.")
        
        for name, page in context.datastore.pages.items():
            _set_edge_energy_value(page, energy.value)
            log.debug(f'Set E0 to value {energy.value} for page {name}')
        
        log.info(f'Set E0 to {energy.value}eV for all pages.')
        return SubCommandResult_Edge_Set()

    def _execute_shift(self, args: SubCommandArguments_Edge_Shift, context: Context) -> SubCommandResult_Edge_Shift:
        log = context.logger.getChild('command.edge.shift')

        log.debug(f'Finding edge energy with shift method in range [{args.range[0]!s}, {args.range[1]!s}], resolution {args.resolution!s}, shift {args.shift!s}, derivative {args.derivative}')

        range = args.range[0].value - args.shift.value, args.range[1].value + args.shift.value
        new_e = _get_shift_axis(range, args.resolution.value)

        log.debug('Preparing interpolator splines for all pages.')
        page_splines = _get_interpolator_splines_for_pages(context.datastore.pages, range, Domain.RECIPROCAL, 'E', 'a', 'ref', log)

        log.debug('Generating interpolated data for all pages.')
        page_interpolated: dict[str, tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]] ={
            name: (spline_a(new_e), spline_ref(new_e)) for name, (spline_a, spline_ref) in page_splines.items()
        }

        log.debug('Calculating shifts for all pages.')
        slide_amount = int(args.shift.value // args.resolution.value)
        compute = partial(_calculate_l2_shift_from_data,
                          derivative=args.derivative,
                          slide_amount=slide_amount,
                          resolution=args.resolution.value,
                          log=log
                        )
        threaded = len(context.datastore.pages) >= 24 and context.options.debug is False
        log.debug(f'Executing shift calculations {"with" if threaded else "without"} threading for {len(context.datastore.pages)} pages.')
        slidenorm = execute_threaded(
            compute,
            page_interpolated,
            argkind='a',
            threaded=threaded,
            pass_key_as='name'
        )

        for name, result in slidenorm.items():
            refE0 = context.datastore.pages[name].meta['refE0']
            edgeE0 = result.x + refE0

            _set_edge_energy_value(context.datastore.pages[name], edgeE0)
            log.debug(f'Set E0 to {edgeE0:0.4f}eV ({result.x:0.4f}eV from reference) for page {name}')
        
        log.info('Calculated edge energies using shift method.')

        return SubCommandResult_Edge_Shift(
            e_range = (args.range[0].value, args.range[1].value),
            results = slidenorm
        )
