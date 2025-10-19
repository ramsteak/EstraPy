from .grammar.transformer import Script
from .core.context import Context

from .commands import execute_command, execute_directive


def execute_script(script: Script, context: Context) -> None:
    for directive in script.directives:
        execute_directive(directive, context)

    for command in script.commands:
        execute_command(command, context)
