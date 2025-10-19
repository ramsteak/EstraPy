from lark import Token


class ParseError(Exception):
    """Exception raised for errors during parsing."""

    def __init__(self, message: str, token: Token | None = None):
        super().__init__(message)
        self.token = token

    def __str__(self) -> str:
        if self.token is not None:
            return f'{self.args[0]} (at line {self.token.line}, column {self.token.column})'
        return self.args[0]


class CommandSyntaxError(ParseError):
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
