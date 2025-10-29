import numpy as np

from dataclasses import dataclass
from lark import Token, Tree
from typing import Self

from ..core.grammarclasses import CommandArguments, Command
from ..core.context import Context, ParseContext
from ..core.number import Number, parse_number, parse_range
from ..grammar.commandparser import CommandArgumentParser
from ..core.misc import parse_edge
    

@dataclass(slots=True)
class SubCommandArguments_Align_Shift(CommandArguments):
    range: tuple[Number, Number]
    resolution: Number
    shift: Number
    derivative: int

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
sub_calc.add_argument('energy', '--energy', '--E0', '-E', type=parse_edge, required=False, default=None)
sub_calc.add_argument('search', '--search', '--sE0', type=parse_edge, required=False, default=None)
sub_calc.add_argument('delta', '--delta', '-d', type=parse_number, required=False, default=None)

sub_shift = CommandArgumentParser(SubCommandArguments_Align_Shift, name='shift')
sub_shift.add_argument('range', types=parse_range, nargs=2, required=False, default=(Number(None, -np.inf, None), Number(None, np.inf, None)))
sub_shift.add_argument('resolution', '--resolution', '--res', type=parse_number, required=False, default=1.0)
sub_shift.add_argument('shift', '--shift', '-s', type=parse_number, required=False, default=0.0)
sub_shift.add_argument('derivative', '--derivative', '--deriv', type=int, required=False, default=1)

parse_align_command = CommandArgumentParser(CommandArguments_Align, name='align')
parse_align_command.add_subparser('calc', sub_calc, 'mode')
parse_align_command.add_subparser('shift', sub_shift, 'mode')


@dataclass(slots=True)
class Command_Align(Command[CommandArguments_Align]):
    @classmethod
    def parse(cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext) -> Self:
        arguments = parse_align_command(commandtoken, tokens, parsecontext)
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> None:
        ...  # Implement the execution logic for the 'align' command here
