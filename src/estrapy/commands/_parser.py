import argparse

from typing import NamedTuple
from ._exceptions import InputFileParsingException
from ._numberunit import NUMBER_UNIT

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
        # Dirty workaround: if the value is of the form NUMBER_UNIT then replace
        # the "-" with "!THISISANEGATIVE!" temporarily. parse_nu can handle this prefix.
        args = [a if not a.startswith("-") else 
                (a if NUMBER_UNIT.match(a) is None else "!THISISANEGATIVE!"+a.removeprefix("-")) for a in args]

        try:
            namespace = self.parse_args(args, _namespace)
        except argparse.ArgumentError as E:
            raise InputFileParsingException(E)

        return namespace  # type: ignore
