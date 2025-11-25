import pytest
import numpy as np

from estrapy.grammar.mathexpressions import Expression

# -------------------------------------------------------------
# Algebraic Tests, should pass

def test_expression_variable():
    expr_str = "a"
    expr = Expression[float].compile(expr_str)
    result = expr(a=1, b=2)
    assert result == 1

def test_expression_zero_literal():
    expr_str = "0"
    expr = Expression[float].compile(expr_str)
    result = expr(a=5, b=3)
    assert result == 0

def test_expression_addition():
    expr_str = "a + b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=1, b=2)
    assert result == 3

def test_expression_subtraction():
    expr_str = "a - b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=5, b=3)
    assert result == 2

def test_expression_multiplication():
    expr_str = "a * b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=4, b=2)
    assert result == 8

def test_expression_division():
    expr_str = "a / b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=8, b=2)
    assert result == 4

def test_expression_power():
    expr_str = "a ** b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=2, b=3)
    assert result == 8

def test_expression_unary_minus():
    expr_str = "-a"
    expr = Expression[float].compile(expr_str)
    result = expr(a=5)
    assert result == -5

def test_expression_parenthesized_group():
    expr_str = "(a + b) * c"
    expr = Expression[float].compile(expr_str)
    result = expr(a=1, b=2, c=3)
    assert result == 9

def test_expression_operator_precedence():
    expr_str = "a + b * c"
    expr = Expression[float].compile(expr_str)
    result = expr(a=1, b=2, c=3)
    assert result == 7

def test_expression_sin():
    expr_str = "sin(a)"
    expr = Expression[float].compile(expr_str)
    result = expr(a=np.pi / 2)
    assert result == 1

def test_expression_cos_plus_sin():
    expr_str = "cos(a) + sin(b)"
    expr = Expression[float].compile(expr_str)
    result = expr(a=0, b=np.pi / 2)
    assert result == 2

def test_expression_exp_times_log():
    expr_str = "exp(a) * log(b)"
    expr = Expression[float].compile(expr_str)
    result = expr(a=1, b=np.e)
    assert result == np.e

def test_expression_sqrt_sum():
    expr_str = "sqrt(a) + sqrt(b)"
    expr = Expression[float].compile(expr_str)
    result = expr(a=4, b=9)
    assert result == 5

def test_expression_abs_difference():
    expr_str = "abs(a - b)"
    expr = Expression[float].compile(expr_str)
    result = expr(a=3, b=7)
    assert result == 4

def test_expression_less_than():
    expr_str = "a < b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=1, b=2)
    assert result == True

def test_expression_less_equal():
    expr_str = "a <= b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=2, b=2)
    assert result == True

def test_expression_greater_than():
    expr_str = "a > b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=3, b=2)
    assert result == True

def test_expression_greater_equal():
    expr_str = "a >= b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=2, b=2)
    assert result == True

def test_expression_equal():
    expr_str = "a == b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=5, b=5)
    assert result == True

def test_expression_not_equal():
    expr_str = "a != b"
    expr = Expression[float].compile(expr_str)
    result = expr(a=5, b=3)
    assert result == True

def test_expression_chained_less_than():
    expr_str = "a < b < c"
    expr = Expression[float].compile(expr_str)
    result = expr(a=1, b=2, c=3)
    assert result == True

def test_expression_chained_less_equal():
    expr_str = "a <= b <= c"
    expr = Expression[float].compile(expr_str)
    result = expr(a=2, b=2, c=3)
    assert result == True

def test_expression_add_zero():
    expr_str = "a + 0"
    expr = Expression[float].compile(expr_str)
    result = expr(a=5)
    assert result == 5

def test_expression_multiply_one():
    expr_str = "a * 1"
    expr = Expression[float].compile(expr_str)
    result = expr(a=7)
    assert result == 7

def test_expression_parentheses_no_effect():
    expr_str = "(a)"
    expr = Expression[float].compile(expr_str)
    result = expr(a=10)
    assert result == 10

def test_expression_sin_zero():
    expr_str = "sin(0)"
    expr = Expression[float].compile(expr_str)
    result = expr()
    assert result == 0

def test_expression_log_one():
    expr_str = "log(1)"
    expr = Expression[float].compile(expr_str)
    result = expr()
    assert result == 0

# -------------------------------------------------------------
# Tests that should raise exceptions

def test_expression_invalid_chained_comparison():
    expr_str = "a < b > c"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)

def test_expression_invalid_double_plus():
    expr_str = "a ++ b"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)
    
def test_expression_invalid_empty_function_call():
    expr_str = "sin()"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)
    
def test_expression_zero_division_runtime():
    expr_str = "a / 0"
    expr = Expression[float].compile(expr_str)
    with pytest.raises(ZeroDivisionError):
        _ = expr(a=5)
    
def test_expression_unknown_function():
    expr_str = "unknown_func(a)"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)
    
def test_expression_incomplete_comparison():
    expr_str = "a < "
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)
    
def test_expression_function_with_boolean_argument():
    expr_str = "sin(a<b)"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)
    
def test_expression_invalid_mixed_comparison_chain():
    expr_str = "a < b != c"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)

# -------------------------------------------------------------
# Tests that return numpy arrays

def test_expression_vector_addition():
    expr_str = "a + b"
    expr = Expression[np.ndarray].compile(expr_str)
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    result = expr(a=a, b=b)
    expected = np.array([5, 7, 9])
    assert np.array_equal(result, expected)

def test_expression_vector_trig_combination():
    expr_str = "sin(a) * cos(b)"
    expr = Expression[np.ndarray].compile(expr_str)
    a = np.array([0, np.pi/2, np.pi])
    b = np.array([0, np.pi/2, np.pi])
    result = expr(a=a, b=b)
    expected = np.array([0, 0, 0])
    assert np.allclose(result, expected)

def test_expression_vector_exp_div_log():
    expr_str = "exp(a) / log(b)"
    expr = Expression[np.ndarray].compile(expr_str)
    a = np.array([0, 1, 2])
    b = np.array([np.e, np.e**2, np.e**3])
    result = expr(a=a, b=b)
    expected = np.array([1, np.e/2, np.e**2/3])
    assert np.allclose(result, expected)

def test_expression_vector_pythag_norm():
    expr_str = "sqrt(a**2 + b**2)"
    expr = Expression[np.ndarray].compile(expr_str)
    a = np.array([3, 5, 8])
    b = np.array([4, 12, 15])
    result = expr(a=a, b=b)
    expected = np.array([5, 13, 17])
    assert np.allclose(result, expected)

def test_expression_vector_abs_difference():
    expr_str = "abs(a - b)"
    expr = Expression[np.ndarray].compile(expr_str)
    a = np.array([1, 4, 7])
    b = np.array([5, 2, 10])
    result = expr(a=a, b=b)
    expected = np.array([4, 2, 3])
    assert np.array_equal(result, expected)

def test_expression_vector_less_than():
    expr_str = "a < b"
    expr = Expression[np.ndarray].compile(expr_str)
    a = np.array([1, 3, 5])
    b = np.array([2, 3, 4])
    result = expr(a=a, b=b)
    expected = np.array([True, False, False])
    assert np.array_equal(result, expected)

def test_expression_vector_greater_equal():
    expr_str = "a >= b"
    expr = Expression[np.ndarray].compile(expr_str)
    a = np.array([3, 4, 5])
    b = np.array([2, 4, 6])
    result = expr(a=a, b=b)
    expected = np.array([True, True, False])
    assert np.array_equal(result, expected)

# -------------------------------------------------------------
# Test expressions that use unavailable functions

def test_expression_unavailable_factorial():
    expr_str = "factorial(a)"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)

def test_expression_unavailable_gamma():
    expr_str = "gamma(a)"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)

def test_expression_unavailable_hypot():
    expr_str = "hypot(a, b)"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)

def test_expression_unavailable_cbrt():
    expr_str = "cbrt(a)"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)

def test_expression_unavailable_sinc():
    expr_str = "sinc(a)"
    with pytest.raises(Exception):
        _ = Expression[float].compile(expr_str)

