import re
import numpy as np
import numpy.typing as npt

from typing import overload, NamedTuple, Iterable


def attempt_number(s: str) -> str | int | float:
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


# def parse_energy(val: str) -> float:
#     m = NUMBER_EV.match(val)
#     if m is None:
#         raise ValueError(f'Could not parse "{val}" as energy.')
#     value, mult, _ = m.groups()
#     return float(value) * SI_multiplier(mult)


class NumberUnit(NamedTuple):
    value: float
    sign: int
    unit: str


def parse_numberunit(val: str) -> NumberUnit:
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

    return NumberUnit(value, sign, unit)


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
