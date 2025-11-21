import numpy as np

from lark import Token, Tree

from dataclasses import dataclass
from typing import Self

from ..core.grammarclasses import CommandArguments, Command
from ..core.context import Context, ParseContext
from ..grammar.commandparser import CommandArgumentParser
from ..core.number import Number, parse_range, Unit
from ..core.datastore import Domain, ColumnDescription, ColumnKind
from ..core.misc import fmt


@dataclass(slots=True)
class CommandArguments_Postedge(CommandArguments):
    range: tuple[Number, Number]
    degree: int
    mode: str = 'division'
    xaxis: str = 'E'
    kweight: float = 0

parse_postedge_command = CommandArgumentParser(CommandArguments_Postedge)
parse_postedge_command.add_argument('range', types=parse_range, nargs=2, required=True)
parse_postedge_command.add_argument('degree', '--degree', '--deg', type=int, default=2)
parse_postedge_command.add_argument(None, '--constant', '-C', action='store_const', dest='degree', const=0, nargs=0)
parse_postedge_command.add_argument(None, '--linear', '-l', action='store_const', dest='degree', const=1, nargs=0)
parse_postedge_command.add_argument(None, '--quadratic', '-q', action='store_const', dest='degree', const=2, nargs=0)
parse_postedge_command.add_argument(None, '--cubic', '-c', action='store_const', dest='degree', const=3, nargs=0)
parse_postedge_command.add_argument('mode', '--mode', '-m', type=str, required=False, default='subtraction')
parse_postedge_command.add_argument(None, '--subtraction', '--sub', '-s', action='store_const', dest='mode', const='subtraction', nargs=0)
parse_postedge_command.add_argument(None, '--division', '--div', '-d', action='store_const', dest='mode', const='division', nargs=0)
parse_postedge_command.add_argument('xaxis', '--xaxis', type=str, required=False, default='E')
parse_postedge_command.add_argument(None, '--E-axis', '-E', action='store_const', dest='xaxis', const='E', nargs=0)
parse_postedge_command.add_argument(None, '--e-axis', '-e', action='store_const', dest='xaxis', const='e', nargs=0)
parse_postedge_command.add_argument(None, '--k-axis', '-k', action='store_const', dest='xaxis', const='k', nargs=0)
parse_postedge_command.add_argument('kweight', '--kweight', type=float, default=0)

@dataclass(slots=True)
class Command_Postedge(Command[CommandArguments_Postedge, None]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_postedge_command(commandtoken, tokens, parsecontext)
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> None:
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
                    new = ColumnDescription(name='a', unit=None, type=ColumnKind.DATA, deps=['a', 'post'], calc=lambda d,j0=J0: d['a'] / d['post'] * J0)
                case _:
                    raise ValueError(f'Invalid mode "{self.args.mode}" for postedge correction.')
            
            domain.add_column('a', new)

            log.debug(f'Applied post-edge polynomial correction with polynomial {fmt.sup.poly([*poly], floatfmt='0.2g')} to page {name}.')

