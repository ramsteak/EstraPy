from lark import Lark, Transformer, Token
from dataclasses import dataclass
from typing import Self

from ._loader import load_grammar

grammar_data = load_grammar("axisindexpos.lark")

plotspec_parser = Lark(
    grammar_data,
    parser='lalr',
    start='start',
    propagate_positions=True,
)

@dataclass(slots=True)
class AxisIndexPosition:
    figurenumber: int
    axisindex: tuple[int, int]
    axissize: tuple[float | None, float | None]

    @classmethod
    def parse(cls: type[Self], spec: str) -> Self:
        tree = plotspec_parser.parse(spec) # pyright: ignore[reportUnknownMemberType]
        transformer = _AxisIndexPositionTransformer()
        fignum, axpos, axsz = transformer.transform(tree)
        return cls(
            figurenumber=fignum,
            axisindex=axpos,
            axissize=axsz,
        )

class _AxisIndexPositionTransformer(Transformer[Token, tuple[int, tuple[int, int], tuple[float | None, float | None]]]):
    def INT(self, token: Token) -> int:
        return int(token)
    def FLOAT(self, token: Token) -> float:
        return float(token)
    def axispos(self, items: list[int]) -> tuple[int, int]:
        return items[0], items[1]
    def xy_both(self, items: list[float]) -> tuple[float | None, float | None]:
        return items[0], items[1]
    def xy_left(self, items: list[float]) -> tuple[float | None, float | None]:
        return items[0], None
    def xy_right(self, items: list[float]) -> tuple[float | None, float | None]:
        return None, items[0]
    def wh_w(self, items: list[float]) -> tuple[float | None, float | None]:
        return items[0], None
    def wh_h(self, items: list[float]) -> tuple[float | None, float | None]:
        return None, items[0]
    def wh_w_h(self, items: list[float]) -> tuple[float | None, float | None]:
        return items[0], items[1]
    def wh_h_w(self, items: list[float]) -> tuple[float | None, float | None]:
        return items[1], items[0]
    
    def figspec(self, items: list[int | tuple[int, int] | tuple[float | None, float | None]]) -> tuple[int, tuple[int, int], tuple[float | None, float | None]]:
        if len(items) == 3:
            fignum, axpos, axsize = items
        elif len(items) == 2:
            fignum, axpos = items
            axsize = (None, None)
        elif len(items) == 1:
            fignum = items[0]
            axpos = (1, 1)
            axsize = (None, None)
        else:
            raise ValueError("Too many items in figspec")
        
        assert isinstance(fignum, int)
        assert isinstance(axpos, tuple)
        axposx, axposy = axpos
        assert isinstance(axposx, int)
        assert isinstance(axposy, int)
        assert isinstance(axsize, tuple)
        axsizex, axsizey = axsize
        if axsizex is not None:
            assert isinstance(axsizex, float)
        if axsizey is not None:
            assert isinstance(axsizey, float)

        return fignum, (axposx, axposy), (axsizex, axsizey)
