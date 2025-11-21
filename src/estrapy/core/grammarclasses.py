from typing import NamedTuple
from lark import Token, Tree
from dataclasses import dataclass
from typing import Self, Generic, TypeVar

from .context import ParseContext, Context


@dataclass(slots=True)
class Directive: ...


@dataclass(slots=True)
class CommandArguments: ...

@dataclass(slots=True)
class CommandResult: ...

_A = TypeVar('_A', bound=CommandArguments, covariant=True)
_R = TypeVar('_R', bound=CommandResult, covariant=True)


@dataclass(slots=True)
class Command(Generic[_A, _R]):
    line: int
    name: str
    args: _A

    @classmethod
    def parse(cls, commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext) -> Self: ...

    def execute(self, context: Context) -> _R: ...


class Script(NamedTuple):
    directives: list[Directive]
    commands: list[Command[CommandArguments, CommandResult]]
