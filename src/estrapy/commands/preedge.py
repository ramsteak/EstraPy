from lark import Token, Tree

from dataclasses import dataclass
from typing import Self

from ..core.grammarclasses import CommandArguments, CommandMetadata, Command
from ..core.context import Context, ParseContext, LocalContext
from ..grammar.commandparser import CommandArgumentParser
from ..core.number import Number, parse_number


@dataclass(slots=True)
class CommandArguments_Preedge(CommandArguments):
    range: tuple[Number, Number]
    degree: int


parse_preedge_command = CommandArgumentParser(CommandArguments_Preedge)
parse_preedge_command.add_argument('range', type=parse_number, nargs=2, required=True)
parse_preedge_command.add_argument('degree', '--degree', '-d', type=int, default=1)
parse_preedge_command.add_argument(None, '--constant', '-C', action='store_const', destination='degree', const=0)
parse_preedge_command.add_argument(None, '--linear', '-l', action='store_const', destination='degree', const=1)
parse_preedge_command.add_argument(None, '--quadratic', '-q', action='store_const', destination='degree', const=2)
parse_preedge_command.add_argument(None, '--cubic', '-c', action='store_const', destination='degree', const=3)

@dataclass(slots=True)
class LocalContext_Preedge(LocalContext):...

@dataclass(slots=True)
class Command_Preedge(Command[CommandArguments_Preedge, LocalContext_Preedge]):
    @classmethod
    def parse(cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext) -> Self:
        arguments = parse_preedge_command.parse(commandtoken, tokens)
        metadata = CommandMetadata(initialize_context=False, finalize_context=False, execution_context=False, execute_with='sequential')
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
            meta=metadata,
        )
    
    async def initialize(self, context: Context) -> None:
        self.local = LocalContext_Preedge()
    
    async def execute_on(self, page: str, context: Context) -> None:
        assert self.local is not None

        pass

    async def finalize(self, context: Context) -> None:
        pass
