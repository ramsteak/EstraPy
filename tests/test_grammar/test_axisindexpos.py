# tests/test_grammar/test_plotspecification_additional.py
import pytest

from estrapy.core.grammar.axisindexpos import (
    plotspec_parser,
    _AxisIndexPositionTransformer, # pyright: ignore[reportPrivateUsage]
    AxisIndexPosition,
)

def parse_transform(spec: str) -> AxisIndexPosition:
    tree = plotspec_parser.parse(spec) # pyright: ignore[reportUnknownMemberType]
    transformer = _AxisIndexPositionTransformer()
    return transformer.transform(tree) # pyright: ignore[reportReturnType]

@pytest.mark.parametrize(
    "spec, exp_fig, exp_index, exp_size",
    [
        ("1", 1, (1, 1), (None, None)),                        # minimal: just figure
        ("2:3.4", 2, (3, 4), (None, None)),                    # figure + axis pos
        ("1:1.2[0.3x0.4]", 1, (1, 2), (0.3, 0.4)),             # xy both
        ("1:1.2[0.5x]", 1, (1, 2), (0.5, None)),               # xy left
        ("1:1.2[x0.6]", 1, (1, 2), (None, 0.6)),               # xy right
        ("1:1.2[w0.7h0.8]", 1, (1, 2), (0.7, 0.8)),            # wh in w then h order
        ("1:1.2[h0.8w0.7]", 1, (1, 2), (0.7, 0.8)),            # wh in h then w order
        ("1:1.2[w0.9]", 1, (1, 2), (0.9, None)),               # only w
        ("1:1.2[h0.9]", 1, (1, 2), (None, 0.9)),               # only h
        ("3:2.4[.5x1e-1]", 3, (2, 4), (0.5, 0.1)),             # floats with . and exponent
    ],
)
def test_valid_plotspecs(spec: str, exp_fig: int, exp_index: tuple[int, int], exp_size: tuple[float | None, float | None]) -> None:
    result = parse_transform(spec)
    assert isinstance(result, AxisIndexPosition)
    assert result.figurenumber == exp_fig
    assert result.axisindex == exp_index
    # allow small float comparisons for safety
    assert result.axissize == exp_size

@pytest.mark.parametrize(
    "spec",
    [
        "1:",                 # missing axispos after colon
        "1:1.",               # incomplete axispos (missing second INT)
        "1:1.2[0.3xw0.4]",    # mixed xy and wh tokens inside size
        "1:1.2[w0.3x0.4]",    # mixed notation (w prefix with x separator)
        "1:1.2[wx]",          # invalid size tokens
        "abc",                # non-integer figure
        "1:1.2[0.3]",         # ambiguous size token (no x/w/h marker) -> invalid
    ],
)
def test_invalid_plotspecs_raise(spec: str) -> None:
    with pytest.raises(Exception):
        parse_transform(spec)
