from lark import Token, Tree

from dataclasses import dataclass
from typing import Self

from ..core.grammarclasses import CommandArguments, Command
from ..core.context import Context, ParseContext
from ..grammar.commandparser import CommandArgumentParser
from ..core.number import Number, parse_range


@dataclass(slots=True)
class CommandArguments_Preedge(CommandArguments):
    range: tuple[Number, Number]
    degree: int
    x: str
    y: str
    n: str


parse_preedge_command = CommandArgumentParser(CommandArguments_Preedge)
parse_preedge_command.add_argument('range', types=parse_range, nargs=2, required=True)
parse_preedge_command.add_argument('degree', '--degree', '-d', type=int, default=1)
parse_preedge_command.add_argument(None, '--constant', '-C', action='store_const', dest='degree', const=0, nargs=0)
parse_preedge_command.add_argument(None, '--linear', '-l', action='store_const', dest='degree', const=1, nargs=0)
parse_preedge_command.add_argument(None, '--quadratic', '-q', action='store_const', dest='degree', const=2, nargs=0)
parse_preedge_command.add_argument(None, '--cubic', '-c', action='store_const', dest='degree', const=3, nargs=0)
parse_preedge_command.add_argument('x', '--xaxiscol', type=str, default='E')
parse_preedge_command.add_argument('y', '--yaxiscol', type=str, default='a')
parse_preedge_command.add_argument('n', '--newyaxiscol', type=str, default='a')


@dataclass(slots=True)
class Command_Preedge(Command[CommandArguments_Preedge]):
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

    def execute_on(self, page: str, context: Context) -> None:
        pass
