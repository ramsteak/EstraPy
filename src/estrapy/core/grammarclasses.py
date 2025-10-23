from typing import NamedTuple, TypeVar, Generic
from dataclasses import dataclass


@dataclass(slots=True)
class Directive: ...


@dataclass(slots=True)
class CommandArguments: ...


@dataclass(slots=True)
class CommandMetadata:
    chainable: bool
    requires_global_context: bool
    cpu_bound: bool


_T = TypeVar('_T', bound=CommandArguments, covariant=True)


@dataclass(slots=True)
class Command(Generic[_T]):
    line: int
    name: str
    args: _T
    meta: CommandMetadata


class Script(NamedTuple):
    directives: list[Directive]
    commands: list[Command[CommandArguments]]
