from dataclasses import dataclass

from ..core.grammarclasses import CommandArguments
from ..core.context import Context
from ..core.number import Number, parse_number
from ..grammar.commandparser import CommandParser, CommandMetadata
from ..core.misc import parse_edge


@dataclass(slots=True)
class CommandArguments_align(CommandArguments):
    mode: str | None = None
    energy: Number | str | None = None
    search: Number | None = None
    delta: Number | None = None


metadata = CommandMetadata(chainable=True, requires_global_context=False, cpu_bound=True)
parse_align_command = CommandParser(CommandArguments_align, metadata)
parse_align_command.add_argument('mode', type=str, required=False, default=None)
parse_align_command.add_argument('energy', '--energy', '--E0', '-E', type=parse_edge, required=False, default=None)
parse_align_command.add_argument('search', '--search', type=parse_edge, required=False, default=None)
parse_align_command.add_argument('delta', '--delta', '-d', type=parse_number, required=False, default=None)


def execute_align_command(
    command: CommandArguments_align, context: Context
) -> None: ...  # Implement the logic for the 'align' command here
