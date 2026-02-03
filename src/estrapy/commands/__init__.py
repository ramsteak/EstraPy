from lark import Token, Tree

from .directives import Directive_define, Directive_clear, execute_directive, Directive_archive, sorted_directives, Directive_title

from ..core.context import Directive, CommandArguments, Command, CommandResult
from ..core.errors import CommandSyntaxError
from ..core.number import parse_number
from ..core.context import ParseContext

__all__ = [
    'parse_directive',
    'parse_command',
    'execute_directive',
    'sorted_directives',
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
        # Directive archive -------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'archive')]:
            return Directive_archive()
        case [Token('COMMANDNAME', 'archive') as d, *_]:
            raise CommandSyntaxError('Invalid archive directive syntax', d)
        # Directive title ---------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'title'), Token('STRING', title)]:
            return Directive_title("".join(title))
        case [Token('COMMANDNAME', 'title'), *tokens]:
            tokens = [t for t in tokens if isinstance(t, Token)]
            # Assume user provided an unquoted string, try to recover
            title: list[str] = []
            position = tokens[0].start_pos if tokens and tokens[0].start_pos else 0
            for token in tokens:
                title.append(" " * ((token.start_pos or 0) - position))
                title.append(str(token.value))
                position = token.end_pos or position + len(str(token.value))
            return Directive_title("".join(title))
        case [Token('COMMANDNAME', 'title') as d]:
            raise CommandSyntaxError('Title directive requires a string argument', d)
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
            from .filein import Command_Filein
            return Command_Filein.parse(t, args, parsecontext)
        # Command align ----------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'align') as t, *args]:
            from .align import Command_Align
            return Command_Align.parse(t, args, parsecontext)
        # Command edge -----------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'edge') as t, *args]:
            from .edge import Command_Edge
            return Command_Edge.parse(t, args, parsecontext)
        # Command noise ----------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'noise') as t, *args]:
            from .noise import Command_Noise
            return Command_Noise.parse(t, args, parsecontext)
        # Command preedge --------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'preedge') as t, *args]:
            from .preedge import Command_Preedge
            return Command_Preedge.parse(t, args, parsecontext)
        # Command postedge -------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'postedge') as t, *args]:
            from .postedge import Command_Postedge
            return Command_Postedge.parse(t, args, parsecontext)
        # Command normalize ------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'normalize') as t, *args]:
            from .normalize import Command_Normalize
            return Command_Normalize.parse(t, args, parsecontext)
        # Command background ------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'background') as t, *args]:
            from .background import Command_Background
            return Command_Background.parse(t, args, parsecontext)
        # Command fourier --------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'fourier') as t, *args]:
            from .fourier import Command_Fourier
            return Command_Fourier.parse(t, args, parsecontext)
        # Command fit ------------------------------------------------------------------------------
        # case [Token('COMMANDNAME', 'fit') as t, *args]:
        #     from .fit import Command_Fit
        #     return Command_Fit.parse(t, args, parsecontext)
        # Command save -----------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'save') as t, *args]:
            from .save import Command_Save
            return Command_Save.parse(t, args, parsecontext)
        # Command cut ------------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'cut') as t, *args]:
            from .cut import Command_Cut
            return Command_Cut.parse(t, args, parsecontext)
        # Commnand average ---------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'average') as t, *args]:
            from .average import Command_Average
            return Command_Average.parse(t, args, parsecontext)
        # Command interpolate ----------------------------------------------------------------------
        case [Token('COMMANDNAME', 'interpolate') as t, *args]:
            from .interpolate import Command_Interpolate
            return Command_Interpolate.parse(t, args, parsecontext)
        # Command show -----------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'show') as t, *args]:
            from .show import Command_Show
            return Command_Show.parse(t, args, parsecontext)
        # Command plot -----------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'plot') as t, *args]:
            from .plot import Command_Plot
            return Command_Plot.parse(t, args, parsecontext)
        # Command deglitch -------------------------------------------------------------------------
        # case [Token('COMMANDNAME', 'deglitch') as t, *args]:
        #     from .deglitch import Command_Deglitch
        #     return Command_Deglitch.parse(t, args, parsecontext)
        # Command exit -----------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'exit' | 'quit') as t, *args]:
            pass # Exit is a statement that has no arguments. Everything after exit is ignored.
            # TODO: verify exit works
        # Unknown command --------------------------------------------------------------------------
        case [Token('COMMANDNAME', str(name)) as c, *args]:
            raise CommandSyntaxError(f"Unknown command '{name}'", c)
        # Invalid command --------------------------------------------------------------------------
        case [Token() as c, *args]:
            raise CommandSyntaxError('Invalid command syntax', c)
        case _:
            raise CommandSyntaxError('Invalid command syntax')

    raise NotImplementedError('Exit command execution is not implemented.')