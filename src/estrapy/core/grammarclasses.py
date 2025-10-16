from typing import NamedTuple, TypeAlias
from dataclasses import dataclass

from ..core.number import Number

@dataclass(slots=True)
class Directive():
    ...

@dataclass(slots=True)
class Command:
    ...

Value: TypeAlias = Number | int | str

class Option(NamedTuple):
    name: str
    values: list[Value]


class Script(NamedTuple):
    directives: list[Directive]
    commands: list[Command]
