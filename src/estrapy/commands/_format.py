from typing import Iterable

ORDINAL = {"1":"st", "2":"nd", "3":"rd"}
CHR = "0123456789+-=()"
SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
SUB = "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"

def th(n:int) -> str:
    """Return the ordinal representation of a number as a string.
    >>> th(1), th(2), th(3), th(4), th(11), th(21)
    ('1st', '2nd', '3rd', '4th', '11th', '21st')
    """
    return f"{n}{ORDINAL.get(str(n)[-1], "th")}"

def plu(n:int, w:str, p:str|None=None) -> str:
    """Return the plural form of a word based on the given number.
    If `p` is provided, it is used as the plural form; otherwise, 's' is appended to `w`.
    >>> plu(1, "cat")
    'cat'
    >>> plu(2, "cat")
    'cats'
    >>> plu(2, "person", "people")
    'people'
    """
    return w if n == 1 else (p or w + "s")

def are(n:int) -> str:
    """Return 'is' if n == 1 else 'are'."""
    return "is" if n == 1 else "are"

def sup(v:int|str) -> str:
    """Convert a string to its superscript representation. Non-supported characters are unchanged.
    >>> sup("32")
    '³²
    """
    return "".join(SUP[CHR.index(c)] if c in CHR else c for c in str(v))

def sub(v:int|str) -> str:
    """Convert a string to its subscript representation. Non-supported characters are unchanged.
    >>> sub("H2O")
    'H₂O'
    """
    return "".join(SUB[CHR.index(c)] if c in CHR else c for c in str(v))

def exp(v:float, ndig:int=1) -> str:
    """Convert a float to a string in scientific notation with superscript exponent.
    The number of digits in the mantissa can be specified with `ndig`.
    >>> exp(12345)
    '1.2x10⁴'
    >>> exp(0.0012345, 2)
    '1.23x10⁻³'
    """
    preformatted = format(v, f"0.{ndig}e")
    if "+" in preformatted or "-" in preformatted:
        val,e = preformatted.split("e")
        while len(e) > 2 and e[1] == "0":
            e = e[0] + e[2:]
        if e == "+0": return f"{val}"
        return f"{val}{sup(e)}"
    else:
        return preformatted

def pol(p:Iterable[float], sep:str=" ") -> str:
    """Convert polynomial coefficients to string.
    Coefficients are in the order of x^0, x^1, ..., x^n
    >>> pol([1,2,3])
    '1x⁰ 2x¹ 3x²'
    >>> pol([1,2,3], sep=" + ")
    '1x⁰ + 2x¹ + 3x²'
    """
    return sep.join(f"{exp(a)}x{sup(e)}" for e,a in enumerate(p))
