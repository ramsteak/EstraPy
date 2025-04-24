import re
import numpy as np
import numpy.typing as npt

from typing import overload, NamedTuple, Iterable
from enum import Enum

def attempt_parse_number(s: str) -> str | int | float:
    if s.isdigit():
        return int(s)
    try:
        return float(s)
    except ValueError:
        return s


SI_PREFIXES = [" kMGTPEZYRQ", " munpfazyrq"]

# NUMBER = r"^([+-]?\d+(?:\.\d+)?)$"
# NUMBER_EV = re.compile(r"^([+-]?\d+(?:\.\d+)?)(?:([kMGTPEZYRQmunpfazyrq])?(eV))?$")
ELEMENT_EDGE = re.compile(
    r"(\w+)\.(\w+)(?:([+-]?\d+(?:\.\d+)?)(?:([kMGTPEZYRQmunpfazyrq])?(eV))?)?"
)
NUMBER_UNIT = re.compile(
    r"^(([+-]?)?\d+(?:\.\d+)?)(?:([kMGTPEZYRQmunpfazyrq])?([A-Za-z]+))?$"
)
# groups: value, sign, multiplier, unit


def SI_multiplier(prefix: str | None) -> int:
    if prefix is None:
        return 1
    if prefix in SI_PREFIXES[0]:
        return 1000 ** (SI_PREFIXES[0].index(prefix))
    if prefix in SI_PREFIXES[1]:
        return 1000 ** (-SI_PREFIXES[1].index(prefix))
    return 1

class NumberUnit(NamedTuple):
    value: float
    sign: int
    unit: str | None

class Bound(Enum):
    INTERNAL = 0
    EXTERNAL = 1

def parse_numberunit(val: str, acceptable_units: Iterable[str|None]|None = None, default_unit:str|None=None) -> NumberUnit:
    m = NUMBER_UNIT.match(val)
    if m is None:
        raise ValueError(f'Could not parse "{val}" as energy.')
    _value, _sign, mult, unit = m.groups()

    value = float(_value) * SI_multiplier(mult)

    if _sign == "+":
        sign = 1
    elif _sign == "-":
        sign = -1
    elif _sign == "":
        sign = 0
    else:
        raise ValueError(f'Invalid sign: "{_sign}"')
    
    if acceptable_units is not None:
        if unit not in acceptable_units:
            _acc = ", ".join(f"{str(u)}" for u in acceptable_units)
            raise ValueError(f'Invalid unit specifier: "{unit}" (accepted units: {_acc})')
        if unit is None:
            unit = default_unit

    return NumberUnit(value, sign, unit)

def parse_numberunit_bound(
    bounds: tuple[str, str],
    acceptable_units: Iterable[str | None] | None = None,
    default_unit: str | None = None,
) -> tuple[NumberUnit | Bound, NumberUnit | Bound]:
    _lower, _upper = bounds

    match _lower:
        case "..": lower = NumberUnit(-np.inf, 0, default_unit)
        case ":.": lower = Bound.EXTERNAL
        case ".:": lower = Bound.INTERNAL
        case _lower:
            try:
                lower = parse_numberunit(_lower, acceptable_units, default_unit)
            except ValueError:
                raise ValueError(f'Invalid lower bound specifier: "{_lower}"')
    match _upper:
        case "..": upper = NumberUnit(+np.inf, 0, default_unit)
        case ":.": upper = Bound.EXTERNAL
        case ".:": upper = Bound.INTERNAL
        case _upper:
            try:
                upper = parse_numberunit(_upper, acceptable_units, default_unit)
            except ValueError:
                raise ValueError(f'Invalid upper bound specifier: "{_upper}"')

    return lower, upper

def parse_numberunit_range(
    bounds: tuple[str, str],
    acceptable_units: Iterable[str | None] | None = None,
    default_unit: str | None = None,
) -> tuple[NumberUnit, NumberUnit]:
    _lower, _upper = parse_numberunit_bound(bounds, acceptable_units, default_unit)

    lower = _lower if isinstance(_lower, NumberUnit) else NumberUnit(-np.inf, 0, default_unit)
    upper = _upper if isinstance(_upper, NumberUnit) else NumberUnit(np.inf, 0, default_unit)
            
    return lower, upper

def parse_edgeenergy(val: str) -> float:
    if (m := NUMBER_UNIT.match(val)) is not None:
        value, _, mult, unit = m.groups()
        if unit not in ("eV", None):
            raise ValueError(f'Invalid energy unit: "{unit}"')
        return float(value) * SI_multiplier(mult)
    elif (m := ELEMENT_EDGE.match(val)) is not None:
        element, edge, value, mult, _ = m.groups()
        shift = float(value) * SI_multiplier(mult) if value is not None else 0

        from xraydb import xray_edge

        energy = xray_edge(element, edge, True)
        return energy + shift  # type: ignore
    else:
        raise ValueError(f'Could not parse "{val}" as edge energy.')

def actualize_bounds(bounds: tuple[NumberUnit|Bound, NumberUnit|Bound], data:Iterable[npt.NDArray], unit:str) -> tuple[NumberUnit, NumberUnit]:
    m = np.array([np.min(d) for d in data])
    M = np.array([np.max(d) for d in data])
    li, le = max(m), min(m)
    ui, ue = min(M), max(M)

    _lower,_upper = bounds
    if _lower == Bound.INTERNAL: lower = NumberUnit(li, 0, unit)
    elif _lower == Bound.EXTERNAL: lower = NumberUnit(le, 0, unit)
    else: lower = _lower
    
    if _upper == Bound.INTERNAL: upper = NumberUnit(ui, 0, unit)
    elif _upper == Bound.EXTERNAL: upper = NumberUnit(ue, 0, unit)
    else: upper = _upper

    return lower,upper
    

# eV to k conversion

KCONST = 0.26246842396479844746511289042048746313978437879180
SCONST = 0.51231672231618445138819837616078387648071784401809  # sqrt(KCONST)
KICNST = 3.8099821109685837311852822702410407881602562961373  # 1 / KCONST


@overload
def E_to_k(v: float, E0: np.floating | float) -> float: ...
@overload
def E_to_k(v: np.floating, E0: np.floating | float) -> np.floating: ...
@overload
def E_to_k(v: npt.NDArray, E0: np.floating | float) -> npt.NDArray: ...
def E_to_k(v, E0: np.floating | float):
    return SCONST * np.sqrt((v - E0))


@overload
def k_to_E(v: float, E0: np.floating | float) -> float: ...
@overload
def k_to_E(v: np.floating, E0: np.floating | float) -> np.floating: ...
@overload
def k_to_E(v: npt.NDArray, E0: np.floating | float) -> npt.NDArray: ...
def k_to_E(v, E0: np.floating | float):
    return v**2 * KICNST + E0


# sk = signed k


@overload
def E_to_sk(v: float, E0: np.floating | float) -> float: ...
@overload
def E_to_sk(v: np.floating, E0: np.floating | float) -> np.floating: ...
@overload
def E_to_sk(v: npt.NDArray, E0: np.floating | float) -> npt.NDArray: ...
def E_to_sk(v, E0: np.floating | float):
    return SCONST * np.sqrt((np.abs(v - E0))) * np.sign(v - E0)


@overload
def sk_to_E(v: float, E0: np.floating | float) -> float: ...
@overload
def sk_to_E(v: np.floating, E0: np.floating | float) -> np.floating: ...
@overload
def sk_to_E(v: npt.NDArray, E0: np.floating | float) -> npt.NDArray: ...
def sk_to_E(v, E0: np.floating | float):
    return v**2 * KICNST * np.sign(v) + E0
