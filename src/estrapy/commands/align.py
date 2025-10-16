from dataclasses import dataclass

from ..core.grammarclasses import Command
from ..core.context import Context
from ..core.number import Number, parse_number
from ..grammar.commandparser import CommandParser

@dataclass(slots=True)
class Command_align(Command):
    mode: str | None = None
    energy: Number | str | None = None
    search: Number | None = None
    delta: Number | None = None


parse_align_command = CommandParser(Command_align)
parse_align_command.add_argument('mode', type=str, required=False, default=None)
parse_align_command.add_argument('energy', '--energy', '--E0', '-E', type=parse_number, required=False, default=None)
parse_align_command.add_argument('search', '--search', type=parse_number, required=False, default=None)
parse_align_command.add_argument('delta', '--delta', '-d', type=parse_number, required=False, default=None)


def execute_align_command(command: Command_align, context: Context) -> None:
    ...  # Implement the logic for the 'align' command here
