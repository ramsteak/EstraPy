from lark import Token, Tree

class CommandError(Exception):
    """Base exception for command-related errors."""

    def __init__(self, message: str, token: Token | Tree[Token] | list[Token | Tree[Token]] | None = None):
        super().__init__(message)
        self.token = token
    
    def __str__(self) -> str:
        match self.token:
            case None:
                # No token information available
                return self.args[0]
            case Token():
                # Single token information available
                return f'{self.args[0]} (at line {self.token.line}, column {self.token.column})'
            case Tree() | list():
                # Tree node information available. Gather all tokens and find the maximum span.
                return f'{self.args[0]}'


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
