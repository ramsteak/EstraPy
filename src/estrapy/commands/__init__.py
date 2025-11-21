from lark import Token, Tree

from .directives import Directive_define, Directive_clear, execute_directive
from .filein import Command_Filein
from .align import Command_Align
from .edge import Command_Edge
from .noise import Command_Noise
from .preedge import Command_Preedge
from .postedge import Command_Postedge
from .normalize import Command_Normalize
from .background import Command_Background
from .fourier import Command_Fourier
from .fit import Command_Fit
from .save import Command_Save
from .cut import Command_Cut

from ..core.grammarclasses import Directive, CommandArguments, Command, CommandResult
from ..core.errors import CommandSyntaxError
from ..core.number import parse_number
from ..core.context import ParseContext

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


def parse_command(command: list[Token | Tree[Token]], parsecontext: ParseContext) -> Command[CommandArguments, CommandResult]:
    match command:
        # Command filein ---------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'filein') as t, *args]:
            return Command_Filein.parse(t, args, parsecontext)
        # Command align ----------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'align') as t, *args]:
            return Command_Align.parse(t, args, parsecontext)
        # Command edge -----------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'edge') as t, *args]:
            return Command_Edge.parse(t, args, parsecontext)
        # Command noise ----------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'noise') as t, *args]:
            return Command_Noise.parse(t, args, parsecontext)
        # Command preedge --------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'preedge') as t, *args]:
            return Command_Preedge.parse(t, args, parsecontext)
        # Command postedge -------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'postedge') as t, *args]:
            return Command_Postedge.parse(t, args, parsecontext)
        # Command normalize ------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'normalize') as t, *args]:
            return Command_Normalize.parse(t, args, parsecontext)
        # Command background ------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'background') as t, *args]:
            return Command_Background.parse(t, args, parsecontext)
        # Command fourier --------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'fourier') as t, *args]:
            return Command_Fourier.parse(t, args, parsecontext)
        # Command fit ------------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'fit') as t, *args]:
            return Command_Fit.parse(t, args, parsecontext)
        # Command save -----------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'save') as t, *args]:
            return Command_Save.parse(t, args, parsecontext)
        # Command cut ------------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'cut') as t, *args]:
            return Command_Cut.parse(t, args, parsecontext)
        # Unknown command --------------------------------------------------------------------------
        case [Token('COMMANDNAME', str(name)) as c, *args]:
            raise CommandSyntaxError(f"Unknown command '{name}'", c)
        # Invalid command --------------------------------------------------------------------------
        case [Token() as c, *args]:
            raise CommandSyntaxError('Invalid command syntax', c)
        case _:
            raise CommandSyntaxError('Invalid command syntax')
