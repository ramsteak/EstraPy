from typing import TypeAlias, Union, Sequence
from lark import Token, Tree

TokenLike: TypeAlias = Union[Token, Tree[Token], Sequence[Union[Token, Tree[Token]]]]

class CommandError(Exception):
    """Base exception for command-related errors."""

    def __init__(self, message: str, token: TokenLike | None = None):
        super().__init__(message)
        self.token = token


class CommandParseError(CommandError):
    """Exception raised for errors during parsing."""


class CommandSyntaxError(CommandParseError):
    """Exception raised for syntax errors in commands."""


class ExecutionError(Exception):
    """Exception raised for errors during command execution."""

    pass


class ArgumentError(Exception):
    """Exception raised for errors related to command arguments."""

    pass


class DuplicateArgumentError(ArgumentError):
    """Exception raised for duplicate command arguments."""

    pass
