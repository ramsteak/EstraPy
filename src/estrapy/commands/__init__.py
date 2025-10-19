from lark import Token, Tree
from ..core.grammarclasses import Option, Directive, Command, Value
from ..core.errors import CommandSyntaxError
from ..core.number import parse_number
from ..core.context import Context, ParseContext

from .directives import Directive_define, Directive_clear, execute_directive

from .filein import parse_filein_command, Command_filein, execute_filein_command
from .align import parse_align_command, Command_align, execute_align_command
from .noise import parse_noise_command, Command_noise, execute_noise_command

__all__ = [
    "parse_directive",
    "parse_command",
    "execute_command",
    "execute_directive",
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
            raise CommandSyntaxError("Invalid define directive syntax", d)
        # Directive clear ---------------------------------------------------------------------------
        case [Token('COMMANDNAME', 'clear')]:
            return Directive_clear()
        case [Token('COMMANDNAME', 'clear') as d, *_]:
            raise CommandSyntaxError("Invalid clear directive syntax", d)
        # Unknown directive -------------------------------------------------------------------------
        case [Token('COMMANDNAME', str(name)) as d, *_]:
            raise CommandSyntaxError(f"Unknown directive '{name}'", d)
        # Invalid directive -------------------------------------------------------------------------
        case [Token() as d, *_]:
            raise CommandSyntaxError("Invalid directive syntax", d)
        case _:
            raise CommandSyntaxError("Invalid directive syntax")

def parse_command_argument(arg: Token | Tree[Token]) -> Option | Value:
    match arg:
        case Token('STRING', value):
            return str(value)
        case Token('INTEGER', value):
            return int(value)
        case Token('FLOAT', value):
            return parse_number(value)
        case Tree(Token('RULE', 'option'), [Token('OPTION', str(name)), *values]):
            vals = [parse_command_argument(v) for v in values]
            vals = [v for v in vals if not isinstance(v, Option)]
            return Option(name, vals)
        case Token() as arg:
            raise CommandSyntaxError("Invalid command argument syntax", arg)
        case Tree(Token() as a, _):
            raise CommandSyntaxError("Invalid command argument syntax", a)
        case _:
            raise CommandSyntaxError("Invalid command argument syntax")
        

def parse_command(command: list[Token | Tree[Token]], parsecontext: ParseContext) -> Command:
    match command:
        # Command filein ---------------------------------------------------------------------------
        case ['filein', *args]:
            return parse_filein_command(args, parsecontext)
        # Command align ----------------------------------------------------------------------------
        case ['align', *args]:
            return parse_align_command(args, parsecontext)
        # Command noise ----------------------------------------------------------------------------
        case ['noise', *args]:
            return parse_noise_command(args, parsecontext)
        # Unknown command --------------------------------------------------------------------------
        case [Token('COMMANDNAME', str(name)) as c, *args]:
            raise CommandSyntaxError(f"Unknown command '{name}'", c)
        # Invalid command --------------------------------------------------------------------------
        case [Token() as c, *args]:
            raise CommandSyntaxError("Invalid command syntax", c)
        case _:
            raise CommandSyntaxError("Invalid command syntax")

def execute_command(command: Command, context: Context) -> None:
    # print(f"Executing command: {command}")
    match command:
        case Command_filein(): # type: ignore
            with context.timers.time("execution/filein"):
                execute_filein_command(command, context) # type: ignore
        case Command_align():
            with context.timers.time("execution/align"):
                execute_align_command(command, context)
        case Command_noise():
            with context.timers.time("execution/noise"):
                execute_noise_command(command, context)
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
