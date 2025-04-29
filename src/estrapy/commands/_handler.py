from abc import ABC, abstractmethod
from dataclasses import dataclass
from shlex import split
from typing import NamedTuple, Any

from ._parser import Token
from ._context import Context, CommandResult


class AbstractCommandHandler(ABC):

    @staticmethod
    @abstractmethod
    def tokenize(lines: list[tuple[int, str]]) -> list[Token]: ...

    @staticmethod
    @abstractmethod
    def parse(tokens: list[Token], context: Context) -> NamedTuple: ...

    @staticmethod
    @abstractmethod
    def execute(args: NamedTuple, context: Context) -> CommandResult: ...

    @staticmethod
    @abstractmethod
    def undo(args: NamedTuple, context: Context) -> CommandResult: ...


class CommandHandler(AbstractCommandHandler):

    @staticmethod
    def tokenize(lines: list[tuple[int, str]]) -> list[Token]:
        return [
            Token(token, lineno, line.index(token))
            for lineno, line in lines
            for token in split(line)
        ]
