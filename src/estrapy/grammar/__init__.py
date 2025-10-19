from importlib.resources import files
from lark import Lark

from .indenter import EstraIndenter
from .transformer import EstraTransformer

__all__ = [
    "file_parser",
    "EstraTransformer",
]

grammar_data = files("estrapy.grammar").joinpath("estrapy.lark").read_text()

file_parser = Lark(
    grammar_data,
    parser="lalr",
    start="start",
    postlex=EstraIndenter(),
    propagate_positions=True,
)

