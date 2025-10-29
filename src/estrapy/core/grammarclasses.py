from typing import NamedTuple
from lark import Token, Tree
from dataclasses import dataclass
from typing import Self, Generic, TypeVar, Any

from .context import ParseContext, Context


@dataclass(slots=True)
class Directive: ...


@dataclass(slots=True)
class CommandArguments: ...


_A = TypeVar('_A', bound=CommandArguments, covariant=True)


@dataclass(slots=True)
class Command(Generic[_A]):
    line: int
    name: str
    args: _A

    @classmethod
    def parse(cls, commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext) -> Self: ...

    def execute(self, context: Context) -> None: ...


class Script(NamedTuple):
    directives: list[Directive]
    commands: list[Command[Any]]
