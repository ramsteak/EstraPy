from .commands import execute_directive, sorted_directives
from .core.context import Script, Context

def execute_script(script: Script, context: Context) -> None:
    with context.timers.time('execution/directives'):
        for directive in sorted_directives(script.directives):
            execute_directive(directive, context)
 
    for command in script.commands:
        with context.timers.time(f'execution/{command.name} (line {command.line})'):
            try:
                res = command.execute(context)
            except Exception as e:
                e.add_note(f"Error executing command '{command.name}' at line {command.line}")
                raise
            
            if res is not None: # pyright: ignore[reportUnnecessaryComparison]
                context.results[command.outname or command.name] = res
