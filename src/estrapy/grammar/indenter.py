from lark.indenter import Indenter


class EstraIndenter(Indenter):
    @property
    def NL_type(self) -> str:
        return "_NL"

    @property
    def INDENT_type(self) -> str:
        return "_INDENT"

    @property
    def DEDENT_type(self) -> str:
        return "_DEDENT"

    @property
    def OPEN_PAREN_types(self) -> list[str]:
        return []

    @property
    def CLOSE_PAREN_types(self) -> list[str]:
        return []

    @property
    def tab_len(self) -> int:
        return 4
