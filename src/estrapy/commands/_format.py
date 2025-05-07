from typing import Iterable

ORDINAL = {"1":"st", "2":"nd", "3":"rd"}
CHR = "0123456789+-=()"
SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
SUB = "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"

def th(n:int) -> str:
    return f"{n}{ORDINAL.get(str(n)[-1], "th")}"

def plu(n:int, w:str, p:str|None=None) -> str:
    return w if n == 1 else (p or w + "s")

def are(n:int) -> str:
    return "is" if n == 1 else "are"

def sup(v:int|str) -> str:
    return "".join(SUP[CHR.index(c)] if c in CHR else c for c in str(v))

def sub(v:int|str) -> str:
    return "".join(SUB[CHR.index(c)] if c in CHR else c for c in str(v))

def exp(v:float, ndig:int=1) -> str:
    preformatted = format(v, f"0.{ndig}e")
    if "+" in preformatted or "-" in preformatted:
        val,e = preformatted.split("e")
        while len(e) > 2 and e[1] == "0":
            e = e[0] + e[2:]
        if e == "+0": return f"{val}"
        return f"{val}{sup(e)}"
    else:
        return preformatted

def pol(p:Iterable) -> str:
    return " ".join(f"{exp(a)}x{sup(e)}" for e,a in enumerate(p))
