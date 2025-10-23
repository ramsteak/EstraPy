from lark import Token, Tree
from ..core.grammarclasses import Directive, CommandArguments, Command
from ..core.errors import CommandSyntaxError
from ..core.number import parse_number
from ..core.context import Context, ParseContext

from .directives import Directive_define, Directive_clear, execute_directive

from .filein import parse_filein_command, CommandArguments_filein, execute_filein_command
from .align import parse_align_command, CommandArguments_align, execute_align_command
from .noise import parse_noise_command, CommandArguments_noise, execute_noise_command

__all__ = [
    'parse_directive',
    'parse_command',
    'execute_command',
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


def parse_command(command: list[Token | Tree[Token]], parsecontext: ParseContext) -> Command[CommandArguments]:
    match command:
        # Command filein ---------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'filein') as t, *args]:
            return parse_filein_command(t, args, parsecontext)
        # Command align ----------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'align') as t, *args]:
            return parse_align_command(t, args, parsecontext)
        # Command noise ----------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'noise') as t, *args]:
            return parse_noise_command(t, args, parsecontext)
        # Unknown command --------------------------------------------------------------------------
        case [Token('COMMANDNAME', str(name)) as c, *args]:
            raise CommandSyntaxError(f"Unknown command '{name}'", c)
        # Invalid command --------------------------------------------------------------------------
        case [Token() as c, *args]:
            raise CommandSyntaxError('Invalid command syntax', c)
        case _:
            raise CommandSyntaxError('Invalid command syntax')


def execute_command(command: Command[CommandArguments], context: Context) -> None:
    # TODO: switch to Process pool for CPU-bound tasks and handle metadata.
    # This is backwards implemented as for now, ignoring all metadata etc..
    match command.args:
        case CommandArguments_filein():  # type: ignore
            with context.timers.time('execution/filein'):
                execute_filein_command(command.args, context)  # type: ignore
        case CommandArguments_align():
            with context.timers.time('execution/align'):
                execute_align_command(command.args, context)
        case CommandArguments_noise():
            with context.timers.time('execution/noise'):
                execute_noise_command(command.args, context)
        # case Command_energy():
        #     execute_energy_command(command, context)
        # case Command_preedge():
        #     execute_preedge_command(command, context)
        # case Command_postedge():
        #     execute_postedge_command(command, context)
        # case Command_background():
        #     execute_background_command(command, context)
        # case Command_normalize():
        #     execute_normalize_command(command, context)
        # case Command_fourier():
        #     execute_fourier_command(command, context)
        # case Command_fit():
        #     execute_fit_command(command, context)
        # case Command_save():
        #     execute_save_command(command, context)
        # case Command_plot():
        #     execute_plot_command(command, context)
        case _:
            ...
    # Implement command execution logic here
