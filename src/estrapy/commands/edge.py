from dataclasses import dataclass
from lark import Token, Tree
from typing import Self

from ..core.grammarclasses import CommandArguments, Command
from ..core.context import Context, ParseContext
from ..core.number import Number, parse_number
from ..grammar.commandparser import CommandArgumentParser
from ..core.misc import parse_edge


@dataclass(slots=True)
class CommandArguments_Edge(CommandArguments):
    mode: str
    energy: Number | str
    search: Number
    delta: Number


parse_edge_command = CommandArgumentParser(CommandArguments_Edge)
parse_edge_command.add_argument('mode', type=str, required=False, default=None)
parse_edge_command.add_argument('energy', '--energy', '--E0', '-E', type=parse_edge, required=False, default=None)
parse_edge_command.add_argument('search', '--search', type=parse_edge, required=False, default=None)
parse_edge_command.add_argument('delta', '--delta', '-d', type=parse_number, required=False, default=None)


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

    def execute_on(
        self, page: str, context: Context
    ) -> None: ...  # Implement the execution logic for the 'edge' command here
