import numpy as np

from lark import Token, Tree

from dataclasses import dataclass
from typing import Self, Literal

from ..core.context import Context, ParseContext, Command, CommandResult
from ..core.commandparser import CommandArgumentParser, CommandArguments, field_arg
from ..core._validators import validate_int_non_negative, validate_float_non_negative, validate_range_unit, validate_option_in
from ..core.number import Number, parse_range, Unit
from ..core.datastore import Domain, ColumnDescription, ColumnKind
from ..core.misc import fmt


@dataclass(slots=True)
class CommandArguments_Postedge(CommandArguments):
    range: tuple[Number, Number] = field_arg(
        position=0,
        types=parse_range,
        nargs=2,
        required=True,
        validate=validate_range_unit(Unit.EV)
    )

    degree: int = field_arg(
        flags=['--degree', '--deg'],
        type=int,
        default=2,
        const_flags={
            '--constant': 0, '-C': 0,
            '--linear': 1, '-l': 1,
            '--quadratic': 2, '-q': 2,
            '--cubic': 3, '-c': 3
        },
        validate=validate_int_non_negative
    )

    mode: Literal['subtraction', 'division'] = field_arg(
        flags=['--mode', '-m'],
        type=str,
        required=False,
        default='division',
        const_flags={
            '--subtraction': 'subtraction',
            '--subtract': 'subtraction',
            '--sub': 'subtraction',
            '-s': 'subtraction',
            '--division': 'division',
            '--divide': 'division',
            '--div': 'division',
            '-d': 'division'
        },
        validate=validate_option_in(['subtraction', 'division'])
    )

    xaxis: Literal['E', 'e', 'k'] = field_arg(
        flags=['--xaxis'],
        type=str,
        required=False,
        default='E',
        const_flags={
            '--E-axis': 'E', '-E': 'E',
            '--e-axis': 'e', '-e': 'e',
            '--k-axis': 'k', '-k': 'k'
        },
        validate=validate_option_in(['E', 'e', 'k'])
    )

    kweight: float = field_arg(
        flags=['--kweight'],
        type=float,
        required=False,
        default=0.0,
        validate=validate_float_non_negative
    )

parse_postedge_command = CommandArgumentParser(CommandArguments_Postedge, 'postedge')

class CommandResult_Postedge(CommandResult):
    ...

@dataclass(slots=True)
class Command_Postedge(Command[CommandArguments_Postedge, CommandResult_Postedge]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_postedge_command.parse(commandtoken, tokens)

        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Postedge:
        log = context.logger.getChild('command.postedge')

        for name, page in context.datastore.pages.items():
            domain = page.domains[Domain.RECIPROCAL]
            # TODO: Check that the columns exist and (if needed) E0 is set

            df = domain.get_columns_data(['E', 'e', 'k', 'a'])
            E0 = page.meta['E0']

            match self.args.range[0]:
                case Number(sign=None, value=value, unit=_) if value == -np.inf:
                    idx_l = np.full(len(df), True, dtype=bool)
                case Number(sign=None, value=val, unit=Unit.EV):
                    idx_l = df['E'] >= val
                case Number(sign=_, value=val, unit=Unit.EV):
                    idx_l = df['e'] >= val
                case Number(sign=_, value=val, unit=Unit.K):
                    idx_l = df['k'] >= val
                case _:
                    raise ValueError(f'Invalid range start "{self.args.range[0]}" for postedge correction.')
            match self.args.range[1]:
                case Number(sign=None, value=value, unit=_) if value == np.inf:
                    idx_u = np.full(len(df), True , dtype=bool)
                case Number(sign=None, value=val, unit=Unit.EV):
                    idx_u = df['E'] <= val
                case Number(sign=_, value=val, unit=Unit.EV):
                    idx_u = df['e'] <= val
                case Number(sign=_, value=val, unit=Unit.K):
                    idx_u = df['k'] <= val
                case _:
                    raise ValueError(f'Invalid range end "{self.args.range[1]}" for postedge correction.')
            
            _region = df[idx_l & idx_u]
            
            match self.args.xaxis, self.args.kweight, self.args.degree:
                case 'E', 0, deg:
                    poly = np.polyfit(_region['E'], _region['a'], deg=deg)
                    post = ColumnDescription(name='post', unit=None, type=ColumnKind.DATA, deps=['E'], calc=lambda d, p=poly: np.polyval(p, d['E']))  # type: ignore
                case 'E', kweight, deg:
                    poly = np.polyfit(_region['E'], _region['a'] * (_region['k'] ** kweight), deg=deg)
                    post = ColumnDescription(name='post', unit=None, type=ColumnKind.DATA, deps=['E', 'k'], calc=lambda d, p=poly, kp=kweight: np.polyval(p, d['E']) / (d['k'] ** kp))  # type: ignore
                case 'e', 0, deg:
                    poly = np.polyfit(_region['e'], _region['a'], deg=deg)
                    post = ColumnDescription(name='post', unit=None, type=ColumnKind.DATA, deps=['e'], calc=lambda d, p=poly: np.polyval(p, d['e']))  # type: ignore
                case 'e', kweight, deg:
                    poly = np.polyfit(_region['e'], _region['a'] * (_region['k'] ** kweight), deg=deg)
                    post = ColumnDescription(name='post', unit=None, type=ColumnKind.DATA, deps=['e', 'k'], calc=lambda d, p=poly, kp=kweight: np.polyval(p, d['e']) / (d['k'] ** kp))  # type: ignore
                case 'k', 0, deg:
                    poly = np.polyfit(_region['k'], _region['a'], deg=deg)
                    post = ColumnDescription(name='post', unit=None, type=ColumnKind.DATA, deps=['k'], calc=lambda d, p=poly: np.polyval(p, d['k']))  # type: ignore
                case 'k', kweight, deg:
                    poly = np.polyfit(_region['k'], _region['a'] * (_region['k'] ** kweight), deg=deg)
                    post = ColumnDescription(name='post', unit=None, type=ColumnKind.DATA, deps=['k'], calc=lambda d, p=poly, kp=kweight: np.polyval(p, d['k']) / (d['k'] ** kp))  # type: ignore
                case _:
                    raise ValueError(f'Invalid combination of xaxis "{self.args.xaxis}" and kweight "{self.args.kweight}" for postedge correction.')                
            
            domain.add_column('post', post)
            
            # Add variable J0 evaluated as the polynomial at E0
            J0 = float(np.polyval(poly, E0)) # type: ignore
            page.meta['J0'] = J0

            match self.args.mode:
                case 'subtraction':
                    new = ColumnDescription(name='a', unit=None, type=ColumnKind.DATA, deps=['a', 'post'], calc=lambda d,j0=J0: (d['a'] - d['post']) + j0)
                case 'division':
                    new = ColumnDescription(name='a', unit=None, type=ColumnKind.DATA, deps=['a', 'post'], calc=lambda d,j0=J0: d['a'] / d['post'] * j0)
                case _:
                    raise ValueError(f'Invalid mode "{self.args.mode}" for postedge correction.')
            
            domain.add_column('a', new)

            log.debug(f'Applied post-edge polynomial correction with polynomial {fmt.sup.poly([*poly], floatfmt='0.2g')} to page {name}.')

        return CommandResult_Postedge()
    