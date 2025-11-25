from estrapy.core.number import parse_number, Unit
import pytest

def test_parse_number_integer_negative():
    result = parse_number("-42")
    assert result.sign == '-'
    assert result.value == pytest.approx(-42.0) # pyright: ignore[reportUnknownMemberType]
    assert result.unit is None

def test_parse_number_integer_positive_sign():
    result = parse_number("+42")
    assert result.sign == '+'
    assert result.value == pytest.approx(42.0) # pyright: ignore[reportUnknownMemberType]
    assert result.unit is None

def test_parse_number_float_no_integer_part():
    result = parse_number(".5eV")
    assert result.sign is None
    assert result.value == pytest.approx(0.5) # pyright: ignore[reportUnknownMemberType]
    assert result.unit == Unit.EV

def test_parse_number_float_no_fraction_part():
    result = parse_number("5.")
    assert result.sign is None
    assert result.value == pytest.approx(5.0) # pyright: ignore[reportUnknownMemberType]
    assert result.unit is None

def test_parse_number_scientific_uppercase_E():
    result = parse_number("1E3")
    assert result.sign is None
    assert result.value == pytest.approx(1000.0) # pyright: ignore[reportUnknownMemberType]
    assert result.unit is None

def test_parse_number_scientific_negative_exponent():
    result = parse_number("1e-3")
    assert result.sign is None
    assert result.value == pytest.approx(0.001) # pyright: ignore[reportUnknownMemberType]
    assert result.unit is None

def test_parse_number_unit_lowercase():
    result = parse_number("3ev")
    assert result.sign is None
    assert result.value == pytest.approx(3.0) # pyright: ignore[reportUnknownMemberType]
    assert result.unit == Unit.EV

def test_parse_number_unit_uppercase():
    result = parse_number("3EV")
    assert result.sign is None
    assert result.value == pytest.approx(3.0) # pyright: ignore[reportUnknownMemberType]
    assert result.unit == Unit.EV

def test_parse_number_unit_proper():
    result = parse_number("3eV")
    assert result.sign is None
    assert result.value == pytest.approx(3.0) # pyright: ignore[reportUnknownMemberType]
    assert result.unit == Unit.EV

def test_parse_number_unit_milli():
    result = parse_number("2meV")
    assert result.sign is None
    assert result.value == pytest.approx(0.002) # pyright: ignore[reportUnknownMemberType]
    assert result.unit == Unit.EV

def test_parse_number_unit_micro():
    result = parse_number("5uA")
    assert result.sign is None
    assert result.value == pytest.approx(0.000005) # pyright: ignore[reportUnknownMemberType]
    assert result.unit == Unit.A

def test_parse_number_unit_angstrom_lowercase():
    result = parse_number("3a")
    assert result.sign is None
    assert result.value == pytest.approx(3.0) # pyright: ignore[reportUnknownMemberType]
    assert result.unit == Unit.A

def test_parse_number_unit_angstrom_lowercase_unicode():
    result = parse_number("3å")
    assert result.sign is None
    assert result.value == pytest.approx(3.0) # pyright: ignore[reportUnknownMemberType]
    assert result.unit == Unit.A

def test_parse_number_unit_angstrom_uppercase():
    result = parse_number("3A")
    assert result.sign is None
    assert result.value == pytest.approx(3.0) # pyright: ignore[reportUnknownMemberType]
    assert result.unit == Unit.A

def test_parse_number_unit_angstrom_uppercase_unicode():
    result = parse_number("3Å")
    assert result.sign is None
    assert result.value == pytest.approx(3.0) # pyright: ignore[reportUnknownMemberType]
    assert result.unit == Unit.A

# ----------------------------
# Invalid number formats
# ----------------------------

def test_parse_number_empty_string_raises():
    with pytest.raises(ValueError):
        parse_number("")

def test_parse_number_only_sign_raises():
    with pytest.raises(ValueError):
        parse_number("-")
    with pytest.raises(ValueError):
        parse_number("+")

def test_parse_number_only_decimal_point_raises():
    with pytest.raises(ValueError):
        parse_number(".")
    with pytest.raises(ValueError):
        parse_number("-.")
    with pytest.raises(ValueError):
        parse_number("+.")

def test_parse_number_multiple_signs_raises():
    with pytest.raises(ValueError):
        parse_number("+-3")
    with pytest.raises(ValueError):
        parse_number("--5")

def test_parse_number_missing_exponent_raises():
    with pytest.raises(ValueError):
        parse_number("1e")
    with pytest.raises(ValueError):
        parse_number("1E")
        
def test_parse_number_exponent_no_base_raises():
    with pytest.raises(ValueError):
        parse_number("e10")
    with pytest.raises(ValueError):
        parse_number("E+5")

def test_parse_number_multiple_decimal_points_raises():
    with pytest.raises(ValueError):
        parse_number("1.2.3")
    with pytest.raises(ValueError):
        parse_number("..5")

# ----------------------------
# Invalid unit formats
# ----------------------------

def test_parse_number_invalid_unit_raises():
    with pytest.raises(ValueError):
        parse_number("5xyz")
    with pytest.raises(ValueError):
        parse_number("3kWatt")  # assuming 'kWatt' is not a recognized unit
    with pytest.raises(ValueError):
        parse_number("2eVV")    # double unit suffix

def test_parse_number_unit_attached_to_non_number_raises():
    with pytest.raises(ValueError):
        parse_number("mV")  # no numeric value
    with pytest.raises(ValueError):
        parse_number("eV")  # missing number

# ----------------------------
# Invalid whitespace or characters
# ----------------------------

def test_parse_number_internal_whitespace_raises():
    with pytest.raises(ValueError):
        parse_number("4 2")
    with pytest.raises(ValueError):
        parse_number("1 .5")
    with pytest.raises(ValueError):
        parse_number("  . 5 ")

def test_parse_number_garbage_characters_raises():
    with pytest.raises(ValueError):
        parse_number("5@")
    with pytest.raises(ValueError):
        parse_number("3.14#")

# ----------------------------
# Edge-case invalid strings
# ----------------------------

def test_parse_number_scientific_invalid_sign_raises():
    with pytest.raises(ValueError):
        parse_number("1e++3")
    with pytest.raises(ValueError):
        parse_number("1e--2")

def test_parse_number_only_unit_with_sign_raises():
    with pytest.raises(ValueError):
        parse_number("-k")
    with pytest.raises(ValueError):
        parse_number("+A")


# ----------------------------
# Valid numbers: SI multipliers + units
# ----------------------------

def test_parse_number_megavolt():
    result = parse_number("1.5MeV")
    assert result.sign is None
    assert result.value == pytest.approx(1.5e6) # pyright: ignore[reportUnknownMemberType]
    assert result.unit == Unit.EV

def test_parse_number_milli_angstrom():
    result = parse_number("2.3mÅ")
    assert result.sign is None
    assert result.value == pytest.approx(0.0023) # pyright: ignore[reportUnknownMemberType]
    assert result.unit == Unit.A

def test_parse_number_kilokelvin():
    result = parse_number("0.5kK")
    assert result.sign is None
    assert result.value == pytest.approx(500.0) # pyright: ignore[reportUnknownMemberType]
    assert result.unit == Unit.K

# ----------------------------
# Valid numbers: unicode units
# ----------------------------

def test_parse_number_micro_angstrom_unicode():
    result = parse_number(".5µA")
    assert result.sign is None
    assert result.value == pytest.approx(0.5e-6) # pyright: ignore[reportUnknownMemberType]
    assert result.unit == Unit.A

def test_parse_number_micro_kelvin_unicode():
    result = parse_number("2μK")
    assert result.sign is None
    assert result.value == pytest.approx(2e-6) # pyright: ignore[reportUnknownMemberType]
    assert result.unit == Unit.K

def test_parse_number_slash_unit():
    result = parse_number("1/Å")
    assert result.sign is None
    assert result.value == pytest.approx(1.0) # pyright: ignore[reportUnknownMemberType]
    assert result.unit == Unit.K

# ----------------------------
# Valid numbers: large/small numbers
# ----------------------------

def test_parse_number_large_giga():
    result = parse_number("9.99G.")
    assert result.sign is None
    assert result.value == pytest.approx(9.99e9) # pyright: ignore[reportUnknownMemberType]
    assert result.unit is None

def test_parse_number_small_micro():
    result = parse_number("-7.2u_")
    assert result.sign == '-'
    assert result.value == pytest.approx(-7.2e-6) # pyright: ignore[reportUnknownMemberType]
    assert result.unit is None

def test_parse_number_very_small_scientific():
    result = parse_number("1e-12")
    assert result.sign is None
    assert result.value == pytest.approx(1e-12) # pyright: ignore[reportUnknownMemberType]
    assert result.unit is None

def test_parse_number_very_large_scientific():
    result = parse_number("1e+12")
    assert result.sign is None
    assert result.value == pytest.approx(1e12) # pyright: ignore[reportUnknownMemberType]
    assert result.unit is None
