import argparse

from typing import NamedTuple
from .exceptions import InputFileParsingException


class Token(NamedTuple):
    value: str
    line: int
    pos: int


class CommandParser(argparse.ArgumentParser):
    def __init__(
        self,
        prog: str | None = None,
        usage: str | None = None,
        description: str | None = None,
        epilog: str | None = None,
        **kwargs
    ) -> None:
        super().__init__(
            prog, usage, description, epilog, **kwargs, exit_on_error=False
        )

    def parse(
        self, tokens: list[Token], _namespace: None | argparse.Namespace = None
    ) -> argparse.Namespace:
        args = [token.value for token in tokens]

        try:
            namespace = self.parse_args(args, _namespace)
        except argparse.ArgumentError:
            raise InputFileParsingException

        return namespace  # type: ignore
