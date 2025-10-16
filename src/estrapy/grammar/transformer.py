from lark import Transformer, Token
from typing import Any


from ..core.grammarclasses import Command, Script, Directive
from ..commands import parse_command, parse_directive


class EstraTransformer(Transformer[Any, Script]):
    def directive(self, items: list[Any]) -> Directive:
        return parse_directive(items)

    def command(self, items: list[Any]) -> Command:
        return parse_command(items)

    def start(self, items: list[Any]) -> Script:
        return Script(
            directives=[item for item in items if isinstance(item, Directive)],
            commands=[item for item in items if isinstance(item, Command)],
        )

    def ESCAPED_STRING(self, item: Token) -> Token:
        # Convert escaped string to normal string by removing quotes and unescaping
        # Escaped strings are always quoted with either " or '
        s = str(item.value)[1:-1]
        s = s.encode('utf-8').decode('unicode_escape')
        return Token('STRING', s, item.start_pos, item.line, item.column, item.end_line, item.end_column, item.end_pos)
