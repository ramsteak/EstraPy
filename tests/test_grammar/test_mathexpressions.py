import pytest
import numpy as np

from estrapy.grammar.mathexpressions import Expression

# -------------------------------------------------------------
# Algebraic Tests, should pass

def test_mathexpression_01():
    expr_str = "a"
    expr = Expression[float].compile(expr_str)
    result = expr(a=1, b=2)
    assert result == 1

def test_mathexpression_02():
    expr_str = "0"
    expr = Expression[float].compile(expr_str)
    result = expr(a=5, b=3)
    assert result == 0

def test_mathexpression_03():
    expr_str = "a + b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=1, b=2)
    assert result == 3

def test_mathexpression_04():
    expr_str = "a - b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=5, b=3)
    assert result == 2

def test_mathexpression_05():
    expr_str = "a * b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=4, b=2)
    assert result == 8

def test_mathexpression_06():
    expr_str = "a / b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=8, b=2)
    assert result == 4

def test_mathexpression_07():
    expr_str = "a ** b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=2, b=3)
    assert result == 8

def test_mathexpression_08():
    expr_str = "-a"
    expr = Expression[float].compile(expr_str)
    result = expr(a=5)
    assert result == -5

def test_mathexpression_09():
    expr_str = "(a + b) * c"
    expr = Expression[float].compile(expr_str)
    result = expr(a=1, b=2, c=3)
    assert result == 9

def test_mathexpression_10():
    expr_str = "a + b * c"
    expr = Expression[float].compile(expr_str)
    result = expr(a=1, b=2, c=3)
    assert result == 7

def test_mathexpression_11():
    expr_str = "sin(a)"
    expr = Expression[float].compile(expr_str)
    result = expr(a=np.pi / 2)
    assert result == 1

def test_mathexpression_12():
    expr_str = "cos(a) + sin(b)"
    expr = Expression[float].compile(expr_str)
    result = expr(a=0, b=np.pi / 2)
    assert result == 2

def test_mathexpression_13():
    expr_str = "exp(a) * log(b)"
    expr = Expression[float].compile(expr_str)
    result = expr(a=1, b=np.e)
    assert result == np.e

def test_mathexpression_14():
    expr_str = "sqrt(a) + sqrt(b)"
    expr = Expression[float].compile(expr_str)
    result = expr(a=4, b=9)
    assert result == 5

def test_mathexpression_15():
    expr_str = "abs(a - b)"
    expr = Expression[float].compile(expr_str)
    result = expr(a=3, b=7)
    assert result == 4

def test_mathexpression_16():
    expr_str = "a < b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=1, b=2)
    assert result == True

def test_mathexpression_17():
    expr_str = "a <= b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=2, b=2)
    assert result == True

def test_mathexpression_18():
    expr_str = "a > b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=3, b=2)
    assert result == True

def test_mathexpression_19():
    expr_str = "a >= b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=2, b=2)
    assert result == True

def test_mathexpression_20():
    expr_str = "a == b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=5, b=5)
    assert result == True

def test_mathexpression_21():
    expr_str = "a != b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=5, b=3)
    assert result == True

def test_mathexpression_22():
    expr_str = "a < b < c"
    expr = Expression[float].compile(expr_str)
    result = expr(a=1, b=2, c=3)
    assert result == True

def test_mathexpression_23():
    expr_str = "a <= b <= c"
    expr = Expression[float].compile(expr_str)
    result = expr(a=2, b=2, c=3)
    assert result == True

def test_mathexpression_24():
    expr_str = "a + 0"
    expr = Expression[float].compile(expr_str)
    result = expr(a=5)
    assert result == 5

def test_mathexpression_25():
    expr_str = "a * 1"
    expr = Expression[float].compile(expr_str)
    result = expr(a=7)
    assert result == 7

def test_mathexpression_26():
    expr_str = "(a)"
    expr = Expression[float].compile(expr_str)
    result = expr(a=10)
    assert result == 10

def test_mathexpression_27():
    expr_str = "sin(0)"
    expr = Expression[float].compile(expr_str)
    result = expr()
    assert result == 0

def test_mathexpression_28():
    expr_str = "log(1)"
    expr = Expression[float].compile(expr_str)
    result = expr()
    assert result == 0

# -------------------------------------------------------------
# Tests that should raise exceptions

def test_mathexpression_29():
    expr_str = "a < b > c"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)

def test_mathexpression_30():
    expr_str = "a ++ b"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)
    
def test_mathexpression_31():
    expr_str = "sin()"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)
    
def test_mathexpression_32():
    expr_str = "a / 0"
    expr = Expression[float].compile(expr_str)
    with pytest.raises(ZeroDivisionError):
        _ = expr(a=5)
    
def test_mathexpression_33():
    expr_str = "unknown_func(a)"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)
    
def test_mathexpression_34():
    expr_str = "a < "
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)
    
def test_mathexpression_35():
    expr_str = "sin(a<b)"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)
    
def test_mathexpression_36():
    expr_str = "a < b != c"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)

# -------------------------------------------------------------
# Tests that return numpy arrays

def test_mathexpression_37():
    expr_str = "a + b"
    expr = Expression[np.ndarray].compile(expr_str)
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    result = expr(a=a, b=b)
    expected = np.array([5, 7, 9])
    assert np.array_equal(result, expected)

def test_mathexpression_38():
    expr_str = "sin(a) * cos(b)"
    expr = Expression[np.ndarray].compile(expr_str)
    a = np.array([0, np.pi/2, np.pi])
    b = np.array([0, np.pi/2, np.pi])
    result = expr(a=a, b=b)
    expected = np.array([0, 0, 0])
    assert np.allclose(result, expected)

def test_mathexpression_39():
    expr_str = "exp(a) / log(b)"
    expr = Expression[np.ndarray].compile(expr_str)
    a = np.array([0, 1, 2])
    b = np.array([np.e, np.e**2, np.e**3])
    result = expr(a=a, b=b)
    expected = np.array([1, np.e/2, np.e**2/3])
    assert np.allclose(result, expected)

def test_mathexpression_40():
    expr_str = "sqrt(a**2 + b**2)"
    expr = Expression[np.ndarray].compile(expr_str)
    a = np.array([3, 5, 8])
    b = np.array([4, 12, 15])
    result = expr(a=a, b=b)
    expected = np.array([5, 13, 17])
    assert np.allclose(result, expected)

def test_mathexpression_41():
    expr_str = "abs(a - b)"
    expr = Expression[np.ndarray].compile(expr_str)
    a = np.array([1, 4, 7])
    b = np.array([5, 2, 10])
    result = expr(a=a, b=b)
    expected = np.array([4, 2, 3])
    assert np.array_equal(result, expected)

def test_mathexpression_42():
    expr_str = "a < b"
    expr = Expression[np.ndarray].compile(expr_str)
    a = np.array([1, 3, 5])
    b = np.array([2, 3, 4])
    result = expr(a=a, b=b)
    expected = np.array([True, False, False])
    assert np.array_equal(result, expected)

def test_mathexpression_43():
    expr_str = "a >= b"
    expr = Expression[np.ndarray].compile(expr_str)
    a = np.array([3, 4, 5])
    b = np.array([2, 4, 6])
    result = expr(a=a, b=b)
    expected = np.array([True, True, False])
    assert np.array_equal(result, expected)

# -------------------------------------------------------------
# Test expressions that use unavailable functions

def test_mathexpression_44():
    expr_str = "factorial(a)"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)

def test_mathexpression_45():
    expr_str = "gamma(a)"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)

def test_mathexpression_46():
    expr_str = "hypot(a, b)"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)

def test_mathexpression_47():
    expr_str = "cbrt(a)"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)

def test_mathexpression_48():
    expr_str = "sinc(a)"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)

