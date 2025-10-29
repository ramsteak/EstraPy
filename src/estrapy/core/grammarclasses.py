from typing import NamedTuple
from lark import Token, Tree
from dataclasses import dataclass
from typing import Self, Generic, TypeVar, Any

from .context import ParseContext, Context

@dataclass(slots=True)
class Directive: ...


@dataclass(slots=True)
class CommandArguments: ...

# A command that has initialize set to False does not require specific context during initialization.
#    e.g. loading static data from another file, setting internal parameters, etc. (or nothing at all).
#    The initialization can therefore be performed anytime before execution
#    Examples are the noise command, where no setup is required.
# A command that has finalize set to False does not require context during finalization.
#    e.g. cleaning up internal data, etc. (or nothing at all).
# A command that has execution set to False does not require the whole context during execution.
#    These commands define execute_on instead of execute, and operate only on local context or data passed to them.

# A command that has initialize set to True requires the full context during initialization.
#    e.g. accessing data from other files, global parameters, etc.
#    The initialization must therefore be performed after all previous commands have been executed.
# A command that has finalize set to True requires the full context during finalization.
# Commands that have execution set to True require the full context during execution
#    e.g. align shift

_A = TypeVar('_A', bound=CommandArguments, covariant=True)

@dataclass(slots=True)
class Command(Generic[_A]):
    line: int
    name: str
    args: _A

    @classmethod
    def parse(cls, commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext) -> Self:
        ...

    def execute(self, context: Context) -> None:
        ...
    


class Script(NamedTuple):
    directives: list[Directive]
    commands: list[Command[Any]]
