import re
import numpy as np
import numpy.typing as npt

from typing import overload, NamedTuple, Iterable
from enum import Enum

from ._numberunit import parse_nu, SI_multiplier

def attempt_parse_number(s: str) -> str | int | float:
    if s.isdigit():
        return int(s)
    try:
        return float(s)
    except ValueError:
        return s
    
_ELEMENT_EDGE = r"(\w+)\.(\w+)(?:([+-]?\d+(?:\.\d+)?)(?:([kMGTPEZYRQmunpfazyrq])?(eV))?)?"
ELEMENT_EDGE = re.compile(_ELEMENT_EDGE)


def parse_edgeenergy(val: str) -> float:
    try:
        nu = parse_nu(val)
        if nu.unit not in ("eV", None):
            raise ValueError(f'Invalid energy unit: "{nu.unit}"')

        return nu.value
    except ValueError:
        m = ELEMENT_EDGE.match(val)

        if m is None: 
            raise ValueError(f'Could not parse "{val}" as edge energy.')
        
        element, edge, value, mult, _ = m.groups()

        shift = float(value) * SI_multiplier(mult) if value is not None else 0

        from xraydb import xray_edge

        energy = xray_edge(element, edge, True)
        return energy + shift  # type: ignore


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
