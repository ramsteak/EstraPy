import numpy as np

from typing import Callable, TypeVar, Collection, Any, Iterable, Union
from enum import Enum

from .number import Number, Unit
from .misc import fuzzy_match_enum, fuzzy_match

def validate_non_null(value: object | None) -> bool:
    """Validator to check if a value is not None."""
    if value is None:
        raise ValueError("Expected a non-null value, got None.")
    return True

def validate_number_unit(*u: Unit) -> Callable[[Number | None], bool]:
    """Validator factory to create a validator that checks if a Number has the specified unit."""
    def validator(number: Number | object | None) -> bool:
        if number is None:
            return True
        
        if not isinstance(number, Number):
            raise ValueError(f"Expected a Number, got {type(number).__name__}.")
        
        if number.unit not in u:
            raise ValueError(f"Expected unit {', '.join(str(_u) for _u in u)}, got {number.unit}.")
        return True
    return validator

def validate_number_positive(number: Number | object | None) -> bool:
    """Validator to check if a Number is positive."""
    if number is None:
        return True
    
    if not isinstance(number, Number):
        raise ValueError(f"Expected a Number, got {type(number).__name__}.")
        
    if number.value <= 0:
        raise ValueError(f"Expected a positive number, got {number.value}.")
    return True

def validate_positive(value: float | int | object | None) -> bool:
    """Validator to check if a numeric value is positive."""
    if value is None:
        return True
    
    if value <= 0:
        raise ValueError(f"Expected a positive value, got {value}.")
    return True

def validate_non_negative(value: float | int | object | None) -> bool:
    """Validator to check if a numeric value is non-negative."""
    if value is None:
        return True
    
    if value < 0:
        raise ValueError(f"Expected a non-negative value, got {value}.")
    return True

def validate_int_positive(value: int | object | None) -> bool:
    """Validator to check if an integer value is positive."""
    if value is None:
        return True
    
    if value <= 0:
        raise ValueError(f"Expected a positive integer, got {value}.")
    return True

def validate_int_non_negative(value: int | object | None) -> bool:
    """Validator to check if an integer value is non-negative."""
    if value is None:
        return True
    
    if value < 0:
        raise ValueError(f"Expected a non-negative integer, got {value}.")
    return True

def validate_float_positive(value: float | object | None) -> bool:
    """Validator to check if a float value is positive."""
    if value is None:
        return True
    
    if value <= 0.0:
        raise ValueError(f"Expected a positive float, got {value}.")
    return True

def validate_float_non_negative(value: float | object | None) -> bool:
    """Validator to check if a float value is non-negative."""
    if value is None:
        return True
    
    if value < 0.0:
        raise ValueError(f"Expected a non-negative float, got {value}.")
    return True


def validate_option_in(options: Collection[Any]) -> Callable[[object | None], bool]:
    """Validator factory to create a validator that checks if a value is in the specified options."""
    def validator(value: str | object | None) -> bool:
        if value is None:
            return True
        
        if value not in options:
            raise ValueError(f"Expected one of {', '.join(str(o) for o in options)}, got {value}.")
        return True
    return validator

def validate_range_unit(*u: Unit) -> Callable[[tuple[Number, Number] | None], bool]:
    """Validator factory to create a validator that checks if a range (tuple of two Numbers) has the specified unit.
    Ranges with infinite values (-np.inf, np.inf) do not require to have a unit (None), but if they do have a unit
    it must be one of the specified units.
    Lower bound must be less than or equal to upper bound. This is checkable only if they have the same unit.
    Only lower bound can be -np.inf, upper bound cannot be -np.inf.
    Only upper bound can be np.inf, lower bound cannot be np.inf.
    """
    def validator(r: tuple[Number, Number] | object | None) -> bool:
        if r is None:
            return True
        
        if not isinstance(r, tuple) or len(r) != 2:
            raise ValueError(f"Expected a tuple of two Numbers, got {r}.")
        
        lower, upper = r
        if not isinstance(lower, Number):
            raise ValueError(f"Expected a tuple of two Numbers, got {r}.")
        if not isinstance(upper, Number):
            raise ValueError(f"Expected a tuple of two Numbers, got {r}.")

        
        if lower.value == -np.inf:
            if lower.unit is not None and lower.unit not in u:
                raise ValueError(f"Expected unit {', '.join(str(_u) for _u in u)} or None for -inf, got {lower.unit}.")
        elif lower.unit not in u:
            raise ValueError(f"Expected unit {', '.join(str(_u) for _u in u)}, got {lower.unit}.")
        
        if upper.value == np.inf:
            if upper.unit is not None and upper.unit not in u:
                raise ValueError(f"Expected unit {', '.join(str(_u) for _u in u)} or None for inf, got {upper.unit}.")
        elif upper.unit not in u:
            raise ValueError(f"Expected unit {', '.join(str(_u) for _u in u)}, got {upper.unit}.")

        if lower.value > upper.value and (lower.unit == upper.unit):
            raise ValueError(f"Expected lower bound less than or equal to upper bound, got {lower.value} > {upper.value}.")
        
        if lower.value == np.inf:
            raise ValueError(f"Expected lower bound not equal to inf, got {lower.value}.")
        if upper.value == -np.inf:
            raise ValueError(f"Expected upper bound not equal to -inf, got {upper.value}.")

        return True
    return validator

_E = TypeVar('_E', bound=Enum)
def type_enum(enum_class: type[_E]) -> Callable[[str], _E]:
    """Validator factory to create a validator that checks if a string is a valid member of the specified enum class."""
    def converter(value: str) -> _E:
        r = fuzzy_match_enum(value, enum_class)
        if r is None:
            raise ValueError(f"Expected one of {', '.join(e.name for e in enum_class)}, got '{value}'.")
        return r
    
    return converter

def type_fuzzy(options: Iterable[Union[str, tuple[str, ...]]], *, min_length: int = 1) -> Callable[[str], str]:
    """Validator factory to create a validator that checks if a string matches one of the specified options."""
    def converter(value: str) -> str:
        return fuzzy_match(value, options, min_length=min_length, strict=True)
    
    return converter

def type_bool(value: str) -> bool:
    """Converter to convert a string to a boolean."""
    value_lower = value.lower()
    if value_lower in ('true', '1', 'yes', 'on', 't', 'y'):
        return True
    elif value_lower in ('false', '0', 'no', 'off', 'f', 'n'):
        return False
    else:
        raise ValueError(f"Expected a boolean value (true/false), got '{value}'.")