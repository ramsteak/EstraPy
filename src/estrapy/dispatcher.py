from .commands import execute_directive, sorted_directives
from .core.grammarclasses import Script
from .core.context import Context

def execute_script(script: Script, context: Context) -> None:
    with context.timers.time('execution/directives'):
        for directive in sorted_directives(script.directives):
            execute_directive(directive, context)
 
    for command in script.commands:
        with context.timers.time(f'execution/{command.name} (line {command.line})'):
            command.execute(context)
