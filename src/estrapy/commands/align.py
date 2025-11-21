import numpy as np

from numpy import typing as npt
from logging import Logger
from scipy.interpolate import make_interp_spline, BSpline # type: ignore missing stub
from dataclasses import dataclass
from lark import Token, Tree
from typing import Self
from functools import partial

from ..core.grammarclasses import CommandArguments, Command, CommandResult
from ..core.context import Context, ParseContext
from ..core.datastore import Domain, DataPage, ColumnDescription, ColumnKind
from ..core.number import Number, parse_number, parse_range, Unit, parse_edge
from ..core.threaded import execute_threaded
from ..grammar.commandparser import CommandArgumentParser
from ..operations.edge_detection import correlation_edge_detection, SlidingL2Result


@dataclass(slots=True)
class SubCommandArguments_Align_Shift(CommandArguments):
    range: tuple[Number, Number]
    resolution: Number
    shift: Number
    derivative: int
    energy: Number | None


@dataclass(slots=True)
class SubCommandArguments_Align_Calc(CommandArguments):
    method: str
    energy: Number
    delta: Number
    search: Number | None


@dataclass(slots=True)
class CommandArguments_Align(CommandArguments):
    mode: SubCommandArguments_Align_Calc | SubCommandArguments_Align_Shift


sub_calc = CommandArgumentParser(SubCommandArguments_Align_Calc, name='calc')
sub_calc.add_argument('method', '--method', '-m', type=str, required=False, default='set')
sub_calc.add_argument('energy', '--energy', '--E0', '-E', type=parse_edge, required=False)
sub_calc.add_argument('search', '--search', '--sE0', type=parse_edge, required=False, default=None)
sub_calc.add_argument('delta', '--delta', '-d', type=parse_number, required=False)

_default_range = (Number(None, -np.inf, None), Number(None, np.inf, None))
sub_shift = CommandArgumentParser(SubCommandArguments_Align_Shift, name='shift')
sub_shift.add_argument('range', types=parse_range, nargs=2, required=False, default=_default_range)
sub_shift.add_argument('resolution', '--resolution', '--res', type=parse_number, required=False, default=Number(None, 0.1, Unit.EV))
sub_shift.add_argument('shift', '--shift', '-s', type=parse_number, required=False, default=Number(None, 5.0, Unit.EV))
sub_shift.add_argument('derivative', '--derivative', '--deriv', type=int, required=False, default=0)
sub_shift.add_argument('energy', '--energy', '--E0', '-E', type=parse_edge, required=False, default=None)

parse_align_command = CommandArgumentParser(CommandArguments_Align, name='align')
parse_align_command.add_subparser('calc', sub_calc, 'mode')
parse_align_command.add_subparser('shift', sub_shift, 'mode')


@dataclass(slots=True)
class SubCommandResult_Align_Shift(CommandResult):
    e_range: tuple[float, float]
    results: dict[str, SlidingL2Result]

CommandResult_Align = SubCommandResult_Align_Shift

def _get_shift_axis(range: tuple[float, float], resolution: float) -> npt.NDArray[np.floating]:
    return np.arange(range[0], range[1] + resolution, resolution)

def _get_interpolator_splines_for_pages(pages: dict[str, DataPage], range: tuple[float, float], dom: Domain, xcol:str, ycol:str, log: Logger) -> dict[str, BSpline]:
    splines: dict[str, BSpline] = {}
    for name, page in pages.items():
        domain = page.domains[dom]
        df = domain.get_columns_data([xcol, ycol])
        index = (df[xcol] >= range[0]) & (df[xcol] <= range[1])

        region = df[index]
        if not region.size:
            log.warning(f'No data in the specified range for page {name}. It will be skipped, this may lead to further errors. Consider adjusting the range.')
            continue

        spline:BSpline = make_interp_spline(region[xcol], region[ycol], k=3) # type: ignore
        spline.extrapolate = True

        splines[name] = spline

    return splines

def _compute_l2_shift_from_data(data: npt.NDArray[np.floating], name: str, reference: npt.NDArray[np.floating], derivative: int, slide_amount: int, resolution: float, log: Logger) -> SlidingL2Result:
    result = correlation_edge_detection(data, reference, derivative, slide_amount, resolution)
    log.debug(f'Calculated shift for page {name}: {result.x:0.4f} eV')
    return result

def _apply_shift_to_page(page: DataPage, edge_energy: float | None, shift_energy: float) -> None:
    if edge_energy is not None:
        page.meta['refE0'] = edge_energy

    domain = page.domains[Domain.RECIPROCAL]
    E_column = ColumnDescription('E', Unit.EV, ColumnKind.AXIS, deps=['E'], calc=lambda df, shift=shift_energy: df['E'] - shift, labl='Energy [eV]')
    domain.add_column('E', E_column)

@dataclass(slots=True)
class Command_Align(Command[CommandArguments_Align, CommandResult_Align]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_align_command(commandtoken, tokens, parsecontext)
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Align:
        match self.args.mode:
            case SubCommandArguments_Align_Calc():
                raise NotImplementedError('Align calc method not implemented yet.')
            
            case SubCommandArguments_Align_Shift():
                return self._execute_shift(self.args.mode, context)
            case _:
                raise NotImplementedError(f"Unknown mode {self.args.mode} in align command.")
    
    def _execute_shift(self, args: SubCommandArguments_Align_Shift, context: Context) -> SubCommandResult_Align_Shift:
        log = context.logger.getChild('command.align.shift')

        log.debug(f'Aligning spectra with correlation method in range [{args.range[0]!s}, {args.range[1]!s}], resolution {args.resolution!s}, shift {args.shift!s}, derivative {args.derivative}, energy {args.energy}')

        range = args.range[0].value - args.shift.value, args.range[1].value + args.shift.value
        new_e = _get_shift_axis(range, args.resolution.value)

        log.debug('Preparing interpolator splines for all pages.')
        page_splines = _get_interpolator_splines_for_pages(context.datastore.pages, range, Domain.RECIPROCAL, 'E', 'ref', log)

        log.debug('Generating interpolated data for all pages.')
        page_interpolated: dict[str, npt.NDArray[np.floating]] = {name: spline(new_e) for name, spline in page_splines.items()}

        log.debug('Calculating average reference spectrum for alignment.')
        average_reference = np.average([*page_interpolated.values()], axis=0)

        slide_amount = int(args.shift.value // args.resolution.value)
        compute = partial(_compute_l2_shift_from_data,
                        reference = average_reference,
                        derivative = args.derivative,
                        slide_amount = slide_amount,
                        resolution = args.resolution.value,
                        log = log
                    )
        threaded = len(context.datastore.pages) >= 24 and context.options.debug is False
        log.debug(f'Executing shift calculations {"with" if threaded else "without"} threading for {len(context.datastore.pages)} pages.')
        slidenorm = execute_threaded(compute, page_interpolated, argkind = 's', threaded = threaded, pass_key_as='name')

        refE0 = args.energy.value if args.energy is not None else None
        for name,result in slidenorm.items():
            _apply_shift_to_page(context.datastore.pages[name], refE0, result.x)
            log.debug(f'Applied shift of {result.x:0.4f} eV to page {name}.')
        
        log.info('Aligned all spectra using correlation method.')

        return SubCommandResult_Align_Shift(
            e_range = (args.range[0].value, args.range[1].value),
            results = slidenorm
        )
