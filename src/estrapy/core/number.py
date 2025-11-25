import re
import numpy as np

from enum import Enum
from dataclasses import dataclass
from unicodedata import normalize

from .edges import edges


class Unit(Enum):
    EV = 'eV'
    A = 'Å'
    K = 'k'


def parse_unit(s: str) -> Unit | None:
    """Parse a string into a Unit enum member. Returns None if the string does not match any unit."""
    # Remove and normalize whitespace and case, removing diacritics
    s = normalize('NFKD', s.strip().casefold()).encode('ascii', 'ignore').decode('ascii')
    match s:
        case 'ev':
            return Unit.EV
        case 'a' | 'ang' | 'angstrom':
            return Unit.A
        case 'k' | 'wavevector' | 'a-1' | '1/angstrom' | 'a^-1' | 'a1':  # result of normalization from a⁻¹
            return Unit.K
        case _:
            raise ValueError(f"Unknown unit: '{s}'")


@dataclass(slots=True, frozen=True)
class Number:
    sign: str | None
    value: float
    unit: Unit | None = None

    def __format__(self, format_spec: str) -> str:
        return f"{self.sign if self.sign == '+' else ''}{self.value:{format_spec}}{self.unit.value if self.unit else ''}"

    def __str__(self) -> str:
        # Add '+' sign for positive numbers. Negative numbers already have '-' sign in str(self.value)
        return f"{self.sign if self.sign == '+' else ''}{self.value}{self.unit.value if self.unit else ''}"

SI_MULTIPLIERS: dict[str, float] = {
    # 'Q': 1e30,  # Useless
    # 'R': 1e27,  # Useless
    # 'Y': 1e24,  # Useless
    # 'Z': 1e21,  # Useless
    # 'E': 1e18,  # Useless
    # 'P': 1e15,  # Useless
    # 'T': 1e12,  # Useless
    'G': 1e9,
    'M': 1e6,
    'k': 1e3,
    'd': 1e-1,
    'c': 1e-2,
    'm': 1e-3,
    'u': 1e-6,  # Useless
    'µ': 1e-6, 'μ': 1e-6, # Micro sign variations
    # 'n': 1e-9,  # Useless
    # 'p': 1e-12, # Useless
    # 'f': 1e-15, # Useless
    # 'a': 1e-18, # Useless
    # 'z': 1e-21, # Useless
    # 'y': 1e-24, # Useless
    # 'r': 1e-27, # Useless
    # 'q': 1e-30, # Useless
}
SI_MULTIPLIER_OPTIONS = ''.join(SI_MULTIPLIERS)
RE_NUMBER = re.compile(
    r'^(([+-])?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)(?:([' + SI_MULTIPLIER_OPTIONS + r'])?([a-zA-ZåÅ/\^⁻¹1-]+))?$'
)
def parse_number(s: str) -> Number:
    """Parse a string into a Number object."""
    s = s.strip()
    if not s:
        raise ValueError('Empty string cannot be parsed as a number.')

    m = RE_NUMBER.match(s)
    if m is None:
        raise ValueError(f"String '{s}' is not a valid number format.")

    num_str, sgn_str, mult_str, unit_str = m.groups()
    mult = SI_MULTIPLIERS.get(mult_str, 1)
    num = float(num_str) * mult
    unit = parse_unit(unit_str) if unit_str else None

    return Number(sgn_str, num, unit)

def try_parse_number(s: str) -> Number | str:
    """Try to parse a string into a Number object. If parsing fails, return the original string."""
    try:
        return parse_number(s)
    except ValueError:
        return s


def parse_edge(edge: str) -> Number:
    """Parse an edge string and return the energy edge in eV.
    The edge string can be shifted by a number.
    Examples:
        - Pd.K -> K edge of Pd
        - Fe.L3+10eV -> L3 edge of Fe shifted by +10 eV
        - Cu.M5-5eV -> M5 edge of Cu shifted by -5 eV
    """
    try:
        return parse_number(edge)
    except ValueError:
        m = re.match(r'^([A-Za-z]+)(?:\.([A-Za-z0-9]+))([+-].*)?', edge)
        if m is None:
            raise ValueError(f"Invalid edge format: '{edge}'")

        element, edge, shift = m.groups()
        if shift is not None:
            shift = parse_number(shift)
            if shift.unit not in (None, Unit.EV):
                raise ValueError('Edge shift must be in eV.')
            shift = shift.value
        else:
            shift = 0.0

        edgename = f'{element}.{edge}'
        if edgename in edges:
            return Number(None, edges[edgename] + shift, Unit.EV)

        # Use xraydb to get the edge energy if everything else fails
        from xraydb import xray_edge  # type: ignore

        try:
            edge_energy: float = xray_edge(element, edge).energy  # type: ignore
            return Number(edge_energy + shift, Unit.EV)  # type: ignore
        except ValueError:
            raise ValueError(f"Invalid element or edge: '{element}.{edge}'")


def parse_range(min_s: str, max_s: str) -> tuple[Number, Number]:
    """Parse two strings into a tuple of Number objects representing a range.
    To specify an open-ended range, use '..' as min_s or max_s."""
    if min_s == '..':
        min_num = Number(None, -np.inf, None)
    else:
        min_num = parse_edge(min_s)

    if max_s == '..':
        max_num = Number(None, np.inf, None)
    else:
        max_num = parse_edge(max_s)

    if min_num.value > max_num.value:
        raise ValueError(f"Invalid range: minimum '{min_s}' is greater than maximum '{max_s}'.")
    return (min_num, max_num)
