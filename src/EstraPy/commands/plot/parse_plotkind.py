# Valid plot identifiers:
# 
# For a x.y plot, the plot can be specified as:
# 
# x(y)              plot(x,y)
# x'(y) dx(y)       plot(x, 1st derivative y)
# x''(y) ddx(y)     plot(x, 2nd derivative y)
# 
# x.y               plot(x,y)
# x.dy              plot(x, 1st derivative y)
# x.ddy             plot(x, 2nd derivative y)
# 
# x can be found inside of df or fd, and y must be in the same dataframe.

import re

from typing import NamedTuple
from matplotlib import colormaps

from ._structs import *

VALID_FUNCCHAR = r"[\w.+*(,)-]"
XY_FUNC_KIND = re.compile(r"^([\w.+*(,)-]+)\(([\w.+*(,)-]+)\)$")
XY_DOT_KIND = re.compile(r"^([\w.+*(,)-]+)\:([\w.+*(,)-]+)$")

TIMES_FUNC = re.compile(r"^\*\(([\w.+*(,)-]+)\)$")
PLUS_FUNC = re.compile(r"^\+\(([\w.+*(,)-]+)\)$")

def split_comma_parens(expr) -> tuple[str, str]:
    depth = 0
    for i, c in enumerate(expr):
        depth += (c == '(') - (c == ')')
        if c == ',' and depth == 0:
            return expr[:i], expr[i+1:]
    raise ValueError("No top-level comma found")

def parse_calc(y:str, x:str) -> CalcToken:

    if y.startswith("d."):
        return Derivative(parse_calc(y.removeprefix("d."), x))
    if y.startswith("d") and len(y) > 3 and y[1].isdecimal() and y[2] == ".":
        # kn, where n is the degree
        n = int(y[1])
        r = parse_calc(y[3:], x)
        for _ in range(n):
            r = Derivative(r)
        return r
    if y.startswith("r."):
        return RealPart(parse_calc(y.removeprefix("r."), x))
    if y.startswith("i."):
        return ImagPart(parse_calc(y.removeprefix("i."), x))
    if y.startswith("a."):
        return Absolute(parse_calc(y.removeprefix("a."), x))
    if y.startswith("p."):
        return ComplexPhase(parse_calc(y.removeprefix("p."), x))
    if y.startswith("k."):
        k = GetColumn(x, "k")
        return Product(k, parse_calc(y.removeprefix("k."), x))
    if y.startswith("k") and len(y) > 3 and y[1].isdecimal() and y[2] == ".":
        # kn, where n is the power
        n = int(y[1])
        r = parse_calc(y[3:], x)
        k = GetColumn(x, "k")
        for _ in range(n):
            r = Product(k, r)
        return r
    if (m:=TIMES_FUNC.match(y)):
        _a,_b = split_comma_parens(m.group(1))
        return Product(parse_calc(_a, x), parse_calc(_b, x))
    if (m:=PLUS_FUNC.match(y)):
        _a,_b = split_comma_parens(m.group(1))
        return Sum(parse_calc(_a, x), parse_calc(_b, x))

    return GetColumn(x, y)


def get_plot_kind(s:str) -> PlotType:
    if (m:=XY_FUNC_KIND.match(s)):
        y,x = m.groups()

        return XYPlot(parse_calc(y,x))
    elif (m:=XY_DOT_KIND.match(s)):
        x,y = m.groups()
        return XYPlot(parse_calc(y,x))
    else:
        raise ValueError(f"Unknown plot type: {s}")