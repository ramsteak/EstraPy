import numpy as np
from numpy import typing as npt
import pandas as pd

from concurrent.futures import ThreadPoolExecutor

from dataclasses import dataclass
from lark import Token, Tree
from typing import Self

from ..core.grammarclasses import CommandArguments, Command
from ..core.context import Context, ParseContext
from ..core.datastore import Domain, DataPage, ColumnDescription, ColumnKind
from ..core.number import Number, parse_number, parse_range, Unit, parse_edge
from ..grammar.commandparser import CommandArgumentParser
from ..operations.edge_detection import correlation_edge_detection

def _set_reference_shift(page: DataPage, edge_energy: float | None, shift_energy: float) -> None:
    if edge_energy is not None:
        page.meta['refE0'] = edge_energy
    domain = page.domains[Domain.RECIPROCAL]
    E_column = ColumnDescription('E', Unit.EV, ColumnKind.AXIS, deps=['E'], calc=lambda df, shift=shift_energy: df['E'] - shift, labl='Energy [eV]')
    domain.add_column('E', E_column)

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
class Command_Align(Command[CommandArguments_Align]):
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

    def execute(self, context: Context) -> None:
        log = context.logger.getChild('command.align')

        match self.args.mode:
            case SubCommandArguments_Align_Calc():
                ...
            case SubCommandArguments_Align_Shift(range, resolution, shift, derivative, energy):
                log.debug(f'Aligning spectra with correlation method in range [{range[0]!s}, {range[1]!s}], resolution {resolution!s}, shift {shift!s}, derivative {derivative}, energy {energy}')
                # TODO: import is costly
                from scipy.interpolate import make_interp_spline, BSpline # type: ignore

                # Prepare data
                min_e, max_e = range[0].value - shift.value, range[1].value + shift.value
                new_e = np.arange(min_e, max_e + resolution.value, resolution.value)
                slide = int(shift.value // resolution.value)

                # Dict of [original data, interpolator spline]
                _data: dict[str, tuple[pd.DataFrame, BSpline]] = {}
                for name, page in context.datastore.pages.items():
                    domain = page.domains[Domain.RECIPROCAL]
                    df = domain.get_columns_data(['E', 'ref'])
                    _region = df[df['E'].between(min_e,max_e)] # type: ignore
                    if not _region.size:
                        log.warning(f'No data in the specified range for page {name}. Skipping.')
                        continue
                    spline:BSpline = make_interp_spline(_region['E'], _region['ref'], k=3) # type: ignore
                    spline.extrapolate = True
                    _data[name] = (_region, spline)

                log.debug('Calculating average reference spectrum for alignment.')
                dat_0:dict[str, npt.NDArray[np.floating]] = {name:spline(new_e) for name, (_, spline) in _data.items()}
                ref_0: npt.NDArray[np.floating] = np.average([*dat_0.values()], axis=0)

                if len(context.datastore.pages) >= 24 and context.options.debug is False:
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        shifts = [executor.submit(correlation_edge_detection, d, ref_0, derivative, slide, resolution.value) for _,d in dat_0.items()]
                    shifts = {name:shift_future.result() for name, shift_future in zip(dat_0.keys(), shifts)}
                else:
                    shifts = {
                        name:correlation_edge_detection(d, ref_0, derivative, slide, resolution.value)
                        for name,d in dat_0.items()
                    }
                
                log.debug('Applying calculated shifts to spectra.')

                for name, page in context.datastore.pages.items():
                    if name not in shifts:
                        continue
                    domain = page.domains[Domain.RECIPROCAL]
                    _set_reference_shift(page, energy.value if energy is not None else None, shifts[name])
                    log.debug(f'Set reference shift to {shifts[name]:0.4f} eV for page {name}')
                    pass
                log.info('Aligned all spectra using correlation method.')
            case _:
                raise NotImplementedError(f"Unknown mode {self.args.mode} in align command.")