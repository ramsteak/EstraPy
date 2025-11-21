from importlib.resources import files, as_file
from lark import Lark

from .indenter import EstraIndenter

with as_file(files("estrapy.grammar") / "estrapy.lark") as grammar_path:
    grammar_data = grammar_path.read_text()


file_parser = Lark(
    grammar_data,
    parser='lalr',
    start='start',
    postlex=EstraIndenter(),
    propagate_positions=True,
)
