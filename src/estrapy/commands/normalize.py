from lark import Token, Tree

from dataclasses import dataclass
from typing import Self

from ..core.context import Command, CommandResult
from ..core.context import Context, ParseContext
from ..core.commandparser import CommandArgumentParser, field_arg, CommandArguments
from ..core.number import Number, try_parse_number
from ..core.datastore import Domain, ColumnDescription, ColumnKind

@dataclass(slots=True)
class CommandArguments_Normalize(CommandArguments):
    factor: Number | str = field_arg(
        flags=['--factor'],
        type=try_parse_number,
        required=False,
        default='J0'
    )

parse_normalize_command = CommandArgumentParser(CommandArguments_Normalize, 'normalize')


class CommandResult_Normalize(CommandResult):
    ...

@dataclass(slots=True)
class Command_Normalize(Command[CommandArguments_Normalize, CommandResult_Normalize]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_normalize_command.parse(commandtoken, tokens)
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Normalize:
        for name, page in context.datastore.pages.items():
            domain = page.domains[Domain.RECIPROCAL]
            match self.args.factor:
                case Number(value=value, unit=None):
                    factor = value
                case str(var) if var in page.meta:
                    factor = float(page.meta[var])
                case _:
                    raise ValueError(f'Cannot normalize page "{name}": Unknown normalization factor "{self.args}')
            
            domain.add_column('mu', ColumnDescription('mu', None, ColumnKind.DATA, ['a'], lambda df,f=factor: df['a'] / f, 'Normalized absorbance'))
            domain.add_column('chi', ColumnDescription('chi', None, ColumnKind.DATA, ['a'], lambda df,f=factor: df['a'] / f - 1, 'EXAFS signal'))
            
        return CommandResult_Normalize()