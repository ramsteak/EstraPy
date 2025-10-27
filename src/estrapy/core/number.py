import re

from enum import Enum
from typing import NamedTuple
from unicodedata import normalize


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
            return None


class Number(NamedTuple):
    sign: str | None
    value: float
    unit: Unit | None = None

    def __format__(self, format_spec: str) -> str:
        return f"{self.value:{format_spec}}{self.unit.value if self.unit else ''}"

    def __str__(self) -> str:
        return f"{self.value}{self.unit.value if self.unit else ''}"


def parse_number(s: str) -> Number:
    """Parse a string into a Number object."""
    s = s.strip()
    if not s:
        raise ValueError('Empty string cannot be parsed as a number.')

    m = re.match(
        r'^(([+-])?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)(?:([GMkdcm])?([a-zA-ZåÅ/\^⁻¹1-]+))?$',
        s,
    )
    if m is None:
        raise ValueError(f"String '{s}' is not a valid number format.")

    num_str, sgn_str, mult_str, unit_str = m.groups()
    mult = {'G': 1e9, 'M': 1e6, 'k': 1e3, 'd': 1e-1, 'c': 1e-2, 'm': 1e-3}.get(mult_str, 1)
    num = float(num_str) * mult
    unit = parse_unit(unit_str) if unit_str else None

    return Number(sgn_str, num, unit)

def parse_range(min_s: str, max_s: str) -> tuple[Number, Number]:
    """Parse two strings into a tuple of Number objects representing a range.
    To specify an open-ended range, use '..' as min_s or max_s."""
    if min_s == '..':
        min_num = Number(None, float('-inf'))
    else:
        min_num = parse_number(min_s)
    
    if max_s == '..':
        max_num = Number(None, float('inf'))
    else:
        max_num = parse_number(max_s)
    
    if min_num.value > max_num.value:
        raise ValueError(f"Invalid range: minimum '{min_s}' is greater than maximum '{max_s}'.")
    return (min_num, max_num)