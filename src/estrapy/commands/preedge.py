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
class CommandArguments_Preedge(CommandArguments):
    range: tuple[Number, Number]
    degree: int

parse_preedge_command = CommandArgumentParser(CommandArguments_Preedge)
parse_preedge_command.add_argument('range', types=parse_range, nargs=2, required=True)
parse_preedge_command.add_argument('degree', '--degree', '-d', type=int, default=1)
parse_preedge_command.add_argument(None, '--constant', '-C', action='store_const', dest='degree', const=0, nargs=0)
parse_preedge_command.add_argument(None, '--linear', '-l', action='store_const', dest='degree', const=1, nargs=0)
parse_preedge_command.add_argument(None, '--quadratic', '-q', action='store_const', dest='degree', const=2, nargs=0)
parse_preedge_command.add_argument(None, '--cubic', '-c', action='store_const', dest='degree', const=3, nargs=0)


@dataclass(slots=True)
class Command_Preedge(Command[CommandArguments_Preedge, None]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_preedge_command(commandtoken, tokens, parsecontext)
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> None:
        log = context.logger.getChild('command.preedge')

        for name, page in context.datastore.pages.items():
            domain = page.domains[Domain.RECIPROCAL]
            # TODO: Check that the columns exist and (if needed) E0 is set
            df = domain.get_columns_data(['E', 'e', 'a'])
            match self.args.range[0]:
                case Number(sign=None, value=val, unit=_) if val == -np.inf:
                    idx_l = np.full(len(df), True, dtype=bool)
                case Number(sign=None, value=val, unit=Unit.EV):
                    idx_l = df['E'] >= val
                case Number(sign=_, value=val, unit=Unit.EV):
                    idx_l = df['e'] >= val
                case Number(sign=_, value=val, unit=Unit.K):
                    idx_l = df['k'] >= val
                case _:
                    raise ValueError(f'Invalid range start "{self.args.range[0]}" for preedge correction.')
            match self.args.range[1]:
                case Number(sign=None, value=val, unit=_) if val == np.inf:
                    idx_u = np.full(len(df), True, dtype=bool)
                case Number(sign=None, value=val, unit=Unit.EV):
                    idx_u = df['E'] <= val
                case Number(sign=_, value=val, unit=Unit.EV):
                    idx_u = df['e'] <= val
                case Number(sign=_, value=val, unit=Unit.K):
                    idx_u = df['k'] <= val
                case _:
                    raise ValueError(f'Invalid range end "{self.args.range[1]}" for preedge correction.')
            
            _region = df[idx_l & idx_u]
            poly = np.polyfit(_region['e'], _region['a'], deg=self.args.degree)
            domain.add_column('pre',
                ColumnDescription(
                    name='pre',
                    unit=None,
                    type=ColumnKind.DATA,
                    deps=['e'],
                    calc=lambda d, p=poly: np.polyval(p, d['e']), # type: ignore
                )
            )
            domain.add_column('a',
                ColumnDescription(
                    name='a',
                    unit=None,
                    type=ColumnKind.DATA,
                    deps=['a', 'pre'],
                    calc=lambda d: d['a'] - d['pre'],
                )
            )
            log.debug(f'Applied pre-edge polynomial subtraction of polynomial {fmt.sup.poly([*poly], floatfmt='0.2g')} to page {name}.')

