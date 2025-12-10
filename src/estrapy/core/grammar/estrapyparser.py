from lark import Lark, Tree, Token
from lark.indenter import Indenter

from ._loader import load_grammar

class EstraIndenter(Indenter):
    @property
    def NL_type(self) -> str:
        return '_NL'

    @property
    def INDENT_type(self) -> str:
        return '_INDENT'

    @property
    def DEDENT_type(self) -> str:
        return '_DEDENT'

    @property
    def OPEN_PAREN_types(self) -> list[str]:
        return []

    @property
    def CLOSE_PAREN_types(self) -> list[str]:
        return []

    @property
    def tab_len(self) -> int:
        return 4

grammar_data = load_grammar("estrapyparser.lark")

file_parser = Lark(
    grammar_data,
    parser='lalr',
    start='start',
    postlex=EstraIndenter(),
    propagate_positions=True,
)

def parse_estrapy_file(filecontent: str) -> Tree[Token]:
    """Parse an EstraPy file content into a parse tree."""
    return file_parser.parse(filecontent) # pyright: ignore[reportUnknownMemberType]
