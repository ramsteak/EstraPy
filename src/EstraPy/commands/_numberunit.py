import re
import numpy as np
import numpy.typing as npt

from typing import NamedTuple, Sequence
from enum import Enum

SI_PREFIXES = ("kMGTPEZYRQ", "munpfazyrq")
SI_MULTIPLIER:dict[str,int] = {
    pname:(1000 ** pval) * (1 if sign == 0 else -1)
    for sign,prefixes in enumerate(SI_PREFIXES)
    for pval,pname in enumerate(prefixes,1)
}

def SI_multiplier(prefix: str | None) -> int:
    if prefix is None: return 1
    return SI_MULTIPLIER.get(prefix, 1)

_ELEMENT_EDGE = r"(\w+)\.(\w+)(?:([+-]?\d+(?:\.\d+)?)(?:([kMGTPEZYRQmunpfazyrq])?(eV))?)?"
ELEMENT_EDGE = re.compile(_ELEMENT_EDGE)

_NUMBER_UNIT = r"^(([+-]?)?\d+(?:\.\d+)?)(?:([kMGTPEZYRQmunpfazyrq])?([A-Za-z]+))?$"
NUMBER_UNIT = re.compile(_NUMBER_UNIT)


class Domain(Enum):
    REAL = "real"
    FOURIER = "fourier"
    PCA = "pca"


class NumberUnit(NamedTuple):
    value: float
    sign: int
    unit: str | None

class Bound(Enum):
    INTERNAL = 0
    EXTERNAL = 1

class NumberUnitRange(NamedTuple):
    lower: NumberUnit | Bound
    upper: NumberUnit | Bound
    inter: NumberUnit | int | None
    domain: Domain | None

def parse_nu(val: str) -> NumberUnit:
    if val.startswith("!THISISANEGATIVE!"):
        return parse_nu("-"+val.removeprefix("!THISISANEGATIVE!"))
    m = NUMBER_UNIT.match(val)
    if m is None:
        raise ValueError(f'Could not parse "{val}".')
    
    _value, _sign, _mult, _unit = m.groups()
    # TODO: TMP:
    # if _unit is None: raise ValueError("Temporary error")
    value = float(_value) * SI_multiplier(_mult)
    sign = ["-", "", "+"].index(_sign) -1

    return NumberUnit(value, sign, _unit)

def get_unit_domain(v:NumberUnit) -> Domain | None:
    if v.unit is None: return None
    return {
        "eV": Domain.REAL,
        "k": Domain.REAL,
        "q": Domain.REAL,
        "A": Domain.FOURIER
    }.get(v.unit)

def _parse_lowbound(val:str) -> tuple[NumberUnit | Bound, Domain | None]:
    match val:
        case "..":
            return NumberUnit(-np.inf, 0, None), None
        case ".:":
            return Bound.INTERNAL, None
        case ":.":
            return Bound.EXTERNAL, None
        case _:
            lower = parse_nu(val)
            return lower, get_unit_domain(lower)
def _parse_higbound(val:str) -> tuple[NumberUnit | Bound, Domain | None]:
    match val:
        case "..":
            return NumberUnit(np.inf, 0, None), None
        case ".:":
            return Bound.EXTERNAL, None
        case ":.":
            return Bound.INTERNAL, None
        case _:
            upper = parse_nu(val)
            return upper, get_unit_domain(upper)

def parse_range(*vals:str, number:int|None=None) -> NumberUnitRange:
    match vals:
        case _lower, _upper:
            lower, ldomain = _parse_lowbound(_lower)
            upper, udomain = _parse_higbound(_upper)

            match lower, upper:
                case NumberUnit(lv,_,u), NumberUnit(uv,us,None) if lv != -np.inf and uv == np.inf:
                    upper = NumberUnit(np.inf, us, u)
                case NumberUnit(lv,ls,None), NumberUnit(uv,_,u) if lv == -np.inf and uv != np.inf:
                    lower = NumberUnit(-np.inf, ls, u)

            match ldomain, udomain:
                case None, None:
                    return NumberUnitRange(lower, upper, number, None)
                case [domain, None] | [None, domain]:
                    return NumberUnitRange(lower, upper, number, domain)
                case ld,ud if ld == ud:
                    return NumberUnitRange(lower, upper, number, ld)
                case _: raise ValueError("Unit domain mismatch.")

        case _lower, _upper, _interval:
            inter = parse_nu(_interval)
            lower, ldomain = _parse_lowbound(_lower)
            upper, udomain = _parse_higbound(_upper)

            if inter.unit == "n":
                # If unit is n, interval is the number of points
                if number is not None: raise ValueError("Two distinct number of points specifications.")

                match ldomain, udomain:
                    case None, None:
                        return NumberUnitRange(lower, upper, int(inter.value), None)
                    case [domain, None] | [None, domain]:
                        return NumberUnitRange(lower, upper, int(inter.value), domain)
                    case ld,ud if ld == ud:
                        return NumberUnitRange(lower, upper, int(inter.value), ld)
                    case _: raise ValueError("Unit domain mismatch.")
            else:
                idomain = get_unit_domain(inter)

                match ldomain, udomain, idomain:
                    case None, None, None:
                        return NumberUnitRange(lower, upper, inter, None)
                    case [domain, None, None] | [None, domain, None] | [None, None, domain]:
                        return NumberUnitRange(lower, upper, inter, domain)
                    case [a,b,None] | [a,None,b] | [None,a,b] if a==b:
                        return NumberUnitRange(lower, upper, inter, a)
                    case ld, ud, id if (ld == ud == id):
                        return NumberUnitRange(lower, upper, inter, id)
                    case _: raise ValueError("Unit domain mismatch.")
    
    raise ValueError("Invalid range.")

def actualize_lower(lowerbound: NumberUnit|Bound, axes:Sequence[npt.NDArray], unit:str|None) -> NumberUnit:
    match lowerbound:
        case Bound.INTERNAL: return NumberUnit(max([np.min(ax) for ax in axes]), 0, unit)
        case Bound.EXTERNAL: return NumberUnit(min([np.min(ax) for ax in axes]), 0, unit)
        case NumberUnit(unit=None): return NumberUnit(lowerbound.value, lowerbound.sign, unit)
        case NumberUnit(): return lowerbound
    raise RuntimeError("Invalid bound.")

def actualize_upper(upperbound: NumberUnit|Bound, axes:Sequence[npt.NDArray], unit:str|None) -> NumberUnit:
    match upperbound:
        case Bound.INTERNAL: return NumberUnit(min([np.max(ax) for ax in axes]), 0, unit)
        case Bound.EXTERNAL: return NumberUnit(max([np.max(ax) for ax in axes]), 0, unit)
        case NumberUnit(unit=None): return NumberUnit(upperbound.value, upperbound.sign, unit)
        case NumberUnit(): return upperbound
    raise RuntimeError("Invalid bound.")

def actualize_range(range:NumberUnitRange, axes:Sequence[npt.NDArray], unit:str|None) -> NumberUnitRange:
    lower, upper = actualize_lower(range.lower, axes, unit), actualize_upper(range.upper, axes, unit)
    return NumberUnitRange(lower, upper, range.inter, range.domain)
