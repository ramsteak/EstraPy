import asyncio

from typing import Sequence, Any

from .grammar.transformer import Script, Command
from .core.context import Context

from .commands import execute_directive

async def execute_script_async(commands: Sequence[Command[Any, Any]], context: Context) -> None:
    for command in commands:
        await command.initialize(context)

        if command.meta.execute_with == 'none':
            await command.execute(context)
        elif command.meta.execute_with == 'sequential':
            for page in context.datastore.pages.keys():
                await command.execute_on(page, context)
        else:
            raise NotImplementedError(f"Execution mode '{command.meta.execute_with}' is not implemented")

        await command.finalize(context)


def execute_script(script: Script, context: Context) -> None:
    # Directives are fast, so we execute them sequentially. They are not performed per-file.
    for directive in script.directives:
        execute_directive(directive, context)

    asyncio.run(execute_script_async(script.commands, context))
