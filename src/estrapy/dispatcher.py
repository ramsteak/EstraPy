from typing import Literal, TypeAlias

from .commands import execute_directive
from .core.grammarclasses import Script
from .core.context import Context

SectionKind: TypeAlias = Literal['none', 'sequential', 'threads', 'processes']


def execute_script(script: Script, context: Context) -> None:
    # Directives are fast, so we execute them sequentially. They are not performed per-file.
    with context.timers.time('execution/directives'):
        for directive in script.directives:
            execute_directive(directive, context)

    for command in script.commands:
        with context.timers.time(f'execution/{command.name} (line {command.line})'):
            command.execute(context)
