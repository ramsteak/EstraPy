import numpy as np
from numpy import typing as npt

from dataclasses import dataclass
from lark import Token, Tree
from typing import Self

from ..core.datastore import Domain, ColumnDescription, ColumnKind, DataPage
from ..core.grammarclasses import CommandArguments, Command
from ..core.context import Context, ParseContext
from ..core.number import Number, parse_number, Unit, parse_range, parse_edge
from ..grammar.commandparser import CommandArgumentParser
from ..operations.edge_detection import correlation_edge_detection

def _set_edge_energy_value(page: DataPage, edge_energy: float, relative: bool = False) -> None:
    from ..operations.axis_conversions import E_to_k

    if relative:
        edge_energy += float(page.meta['refE0'])

    page.meta['E0'] = edge_energy
    domain = page.domains[Domain.RECIPROCAL]
    e_column = ColumnDescription('e', Unit.EV, ColumnKind.AXIS, deps=['E'], calc=lambda df, E0=edge_energy: df['E'] - E0, labl='Relative energy [eV]')
    k_column = ColumnDescription('k', Unit.A, ColumnKind.AXIS, deps=['E'], calc=lambda df, E0=edge_energy: E_to_k(df['E'], E0), labl='Wave vector k [1/Å]')
    domain.add_column('e', e_column)
    domain.add_column('k', k_column)


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
    energy: Number | str


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
class Command_Edge(Command[CommandArguments_Edge]):
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

    def execute(self, context: Context) -> None:
        log = context.logger.getChild('command.edge')

        match self.args.mode:
            case SubCommandArguments_Edge_Set(Number(sign, value, unit) as energy):
                if not (unit is None or unit == Unit.EV):
                    raise ValueError("Only eV unit is supported for E0 setting.")
                if sign is not None:
                    raise ValueError("Relative E0 setting is not supported.")
                
                for name, page in context.datastore.pages.items():
                    _set_edge_energy_value(page, value, relative=False)
                    log.debug(f'Set E0 to value {energy} for page {name}')

            case SubCommandArguments_Edge_Calc():
                ...

            case SubCommandArguments_Edge_Shift(range, resolution, shift, derivative):
                from scipy.interpolate import make_interp_spline, BSpline # type: ignore

                min_e, max_e = range[0].value - shift.value, range[1].value + shift.value
                new_e = np.arange(min_e, max_e + resolution.value, resolution.value)
                slide = int(shift.value // resolution.value)

                # TODO: Parallelize over pages
                for name, page in context.datastore.pages.items():
                    domain = page.domains[Domain.RECIPROCAL]

                    df = domain.get_columns_data(['E', 'ref', 'a'])
                    _region = df[df['E'].between(min_e,max_e)] # type: ignore
                    if not _region.size:
                        log.warning(f'No data in the specified range for page {name}. Skipping.')
                        continue
                
                    # Interpolate 'a' and 'ref' to new energy grid
                    spline_a:BSpline = make_interp_spline(_region['E'].values, _region['a'].values, k=3) # type: ignore
                    spline_ref:BSpline = make_interp_spline(_region['E'].values, _region['ref'].values, k=3) # type: ignore
                    spline_a.extrapolate = True
                    spline_ref.extrapolate = True

                    a: npt.NDArray[np.float64] = spline_a(new_e) # type: ignore
                    ref: npt.NDArray[np.float64] = spline_ref(new_e) # type: ignore

                    difference = correlation_edge_detection(a, ref, d=derivative, slide=slide, dx=resolution.value) # type: ignore
                    # Shift represents the difference between the data E0 and the reference E0
                    
                    _set_edge_energy_value(page, difference, relative=True)
                    refE0 = page.meta['refE0']
                    log.debug(f'Set E0 to {refE0 + difference:0.4f}eV ({difference:0.4f}eV from reference) for page {name}')
            case _:
                raise NotImplementedError(f"Unknown mode {self.args.mode} in edge command.")
        ...
