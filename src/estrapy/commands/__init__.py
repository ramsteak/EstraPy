from lark import Token, Tree
from ..core.grammarclasses import Directive, CommandArguments, Command, LocalContext
from ..core.errors import CommandSyntaxError
from ..core.number import parse_number
from ..core.context import ParseContext

from .directives import Directive_define, Directive_clear, execute_directive

from .filein import Command_Filein
from .align import Command_Align
from .noise import Command_Noise

__all__ = [
    'parse_directive',
    'parse_command',
    'execute_directive',
]


def parse_directive(directive: list[Token | Tree[Token]], parsecontext: ParseContext) -> Directive:
    match directive:
        # Directive define --------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'define'), Token('STRING', name), Token('INTEGER', value)]:
            return Directive_define(name, int(value))
        case [Token('COMMANDNAME', 'define'), Token('STRING', name), Token('FLOAT', value)]:
            return Directive_define(name, parse_number(value))
        case [Token('COMMANDNAME', 'define'), Token('STRING', name), Token('STRING', value)]:
            return Directive_define(name, str(value))
        case [Token('COMMANDNAME', 'define') as d, *_]:
            raise CommandSyntaxError('Invalid define directive syntax', d)
        # Directive clear ---------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'clear')]:
            return Directive_clear()
        case [Token('COMMANDNAME', 'clear') as d, *_]:
            raise CommandSyntaxError('Invalid clear directive syntax', d)
        # Unknown directive -------------------------------------------------------------------------
        case [Token('COMMANDNAME', str(name)) as d, *_]:
            raise CommandSyntaxError(f"Unknown directive '{name}'", d)
        # Invalid directive -------------------------------------------------------------------------
        case [Token() as d, *_]:
            raise CommandSyntaxError('Invalid directive syntax', d)
        case _:
            raise CommandSyntaxError('Invalid directive syntax')


def parse_command(command: list[Token | Tree[Token]], parsecontext: ParseContext) -> Command[CommandArguments, LocalContext]:
    match command:
        # Command filein ---------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'filein') as t, *args]:
            return Command_Filein.parse(t, args, parsecontext)
        # Command align ----------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'align') as t, *args]:
            return Command_Align.parse(t, args, parsecontext)
        # Command noise ----------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'noise') as t, *args]:
            return Command_Noise.parse(t, args, parsecontext)
        # Unknown command --------------------------------------------------------------------------
        case [Token('COMMANDNAME', str(name)) as c, *args]:
            raise CommandSyntaxError(f"Unknown command '{name}'", c)
        # Invalid command --------------------------------------------------------------------------
        case [Token() as c, *args]:
            raise CommandSyntaxError('Invalid command syntax', c)
        case _:
            raise CommandSyntaxError('Invalid command syntax')
