import re

from lark import Transformer, Token
from typing import Any


from .core.context import Command, Script, Directive, CommandArguments, CommandResult
from .core.context import ParseContext
from .commands import parse_command, parse_directive

def safe_unescape(s: str) -> str:
    escape_map = {
        'n': '\n',
        't': '\t',
        'r': '\r',
        '\\': '\\',
        '"': '"',
        "'": "'",
    }

    pattern = r'\\([ntr\\\'"])'
    def _replace(match: re.Match[str]) -> str:
        char_code = match.group(1)
        return escape_map[char_code]
        
    return re.sub(pattern, _replace, s)


class EstraTransformer(Transformer[Any, Script]):
    def __init__(self, parsecontext: ParseContext, visit_tokens: bool = True) -> None:
        super().__init__(visit_tokens)
        self.parsecontext = parsecontext

    def directive(self, items: list[Any]) -> Directive:
        return parse_directive(items, self.parsecontext)

    def command(self, items: list[Any]) -> Command[CommandArguments, CommandResult]:
        return parse_command(items, self.parsecontext)

    def start(self, items: list[Any]) -> Script:
        return Script(
            directives=[item for item in items if isinstance(item, Directive)],
            commands=[item for item in items if isinstance(item, Command)],
        )
    
    def ESCAPED_STRING(self, item: Token) -> Token:
        # Convert escaped string to normal string by removing quotes and unescaping
        # Escaped strings are always quoted with either " or '
        s = safe_unescape(str(item.value)[1:-1])
        # s = s.encode('latin-1', 'backslashreplace').decode('unicode_escape')
        return Token('STRING', s, item.start_pos, item.line, item.column, item.end_line, item.end_column, item.end_pos)
