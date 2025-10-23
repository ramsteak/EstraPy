import asyncio
import inspect

from typing import Sequence

from .grammar.transformer import Script, Command, CommandArguments
from .core.context import Context

from .commands import execute_command, execute_directive

async def execute_script_async(commands: Sequence[Command[CommandArguments]], context: Context) -> None:
    for command in commands:
        maybe_coro = execute_command(command, context)
        if inspect.iscoroutine(maybe_coro):
            await maybe_coro # type: ignore # TODO remove when these will be coroutines


def execute_script(script: Script, context: Context) -> None:
    # Directives are fast, so we execute them sequentially. They are not performed per-file.
    for directive in script.directives:
        execute_directive(directive, context)

    asyncio.run(execute_script_async(script.commands, context))
