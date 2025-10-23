from dataclasses import dataclass
from lark import Token, Tree
from typing import Self

from ..core.grammarclasses import CommandArguments, CommandMetadata, Command
from ..core.context import Context, ParseContext, LocalContext
from ..core.number import Number, parse_number
from ..grammar.commandparser import CommandArgumentParser
from ..core.misc import parse_edge


@dataclass(slots=True)
class CommandArguments_Align(CommandArguments):
    mode: str
    energy: Number | str
    search: Number
    delta: Number

parse_align_command = CommandArgumentParser(CommandArguments_Align)
parse_align_command.add_argument('mode', type=str, required=False, default=None)
parse_align_command.add_argument('energy', '--energy', '--E0', '-E', type=parse_edge, required=False, default=None)
parse_align_command.add_argument('search', '--search', type=parse_edge, required=False, default=None)
parse_align_command.add_argument('delta', '--delta', '-d', type=parse_number, required=False, default=None)

@dataclass(slots=True)
class LocalContext_Align(LocalContext):
    ...

@dataclass(slots=True)
class Command_Align(Command[CommandArguments_Align, LocalContext_Align]):
    @classmethod
    def parse(cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext) -> Self:
        arguments = parse_align_command.parse(commandtoken, tokens)
        metadata = CommandMetadata(initialize_context=False, finalize_context=False, execution_context=False, execute_with='sequential')
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
            meta=metadata,
            local=None,
        )

    async def initialize(self, context: Context):
        pass

    async def execute_on(self, page: str, context: Context) -> None:
        ...  # Implement the execution logic for the 'align' command here
    
    async def finalize(self, context: Context) -> None:
        pass
