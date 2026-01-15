import numpy as np
from numpy import typing as npt
from lark import Lark, Transformer, Token, Tree
from dataclasses import dataclass
from typing import Any, Self, Protocol, Generic, TypeVar

from ._loader import load_grammar

ALLOWED_FUNCTIONS = {
    'sin', 'cos', 'tan',
    'arcsin', 'arccos', 'arctan',
    'sinh', 'cosh', 'tanh',
    'exp', 'log', 'log10', 'log2',
    'sqrt', 'abs', 'real', 'imag',
}

grammar_data = load_grammar("mathexpression.lark")

mathexpression_parser = Lark(
    grammar_data,
    parser='lalr',
    start='start',
    propagate_positions=True,
)

class MathExpressionCompiler(Transformer[Token, str]):
    """This transformer compiles a mathematical expression parse tree into a Python expression string."""
    def number(self, n: list[Token]) -> str:
        return n[0].value
    def infinity(self, n: list[Token]) -> str:
        return "np.inf"
    def var(self, n: list[Token]) -> str:
        return f"{n[0].value}"
    def evar(self, n: list[Token]) -> str:
        # Replace dots with underscores for escaped vars
        return f"{n[0].value.replace('.', '_')}"
    
    def add(self, args: list[Token | str]):
        return f"({args[0]} + {args[1]})"
    def sub(self, args: list[Token | str]):
        return f"({args[0]} - {args[1]})"
    
    def mul(self, args: list[Token | str]):
        return f"({args[0]} * {args[1]})"
    def div(self, args: list[Token | str]):
        return f"({args[0]} / {args[1]})"
    
    def pow(self, args: list[Token | str]):
        return f"({args[0]} ** {args[1]})"
    def neg(self, args: list[Token | str]):
        return f"(-{args[0]})"
    
    def lt(self, _):
        return "<"
    def le(self, _):
        return "<="
    def gt(self, _):
        return ">"
    def ge(self, _):
        return ">="
    def eq(self, _):
        return "=="
    def ne(self, _):
        return "!="
    def compchain1(self, args: list[Token | str]):
        left, op, right = args  # op is a string
        return f"({left} {op} {right})"

    # chain2: a <= b < c
    def compchain2(self, args: list[Token | str]):
        left, op1, mid, op2, right = args
        if op1 in ("==", "!=",) or op2 in ("==", "!=",):
            raise ValueError("Chained equality/inequality comparisons are not supported.")
        if (op1 in ("<", "<=") and op2 in (">", ">=")) or (op1 in (">", ">=") and op2 in ("<", "<=")):
            raise ValueError("Incompatible chained comparisons.")
        # Convert into: (left op1 mid) and (mid op2 right)
        return f"(({left} {op1} {mid}) & ({mid} {op2} {right}))"

    
    def func(self, args: list[Token | str]):
        fname = args[0]
        if fname not in ALLOWED_FUNCTIONS:
            raise ValueError(f"Function '{fname}' is not allowed in mathematical expressions.")
        arglist = ", ".join(args[1:])
        return f"np.{fname}({arglist})"
    def args(self, args: list[Token]):
        return args

@dataclass
class ExprInfo:
    s: str          # rendered string
    prec: int       # precedence level (higher = binds tighter)
    assoc: str=''   # 'L', 'R', or '' for atoms

def _mathexpression_paren_wrap(child: ExprInfo, parent_prec: int, side: str, parent_assoc: str):
    """
    side: 'L' or 'R' — which side child is on in a binary op
    parent_assoc: 'L' or 'R'
    """
    if child.prec > parent_prec:
        return child.s

    if child.prec == parent_prec:
        # same precedence: only add parentheses if associativity breaks
        if parent_assoc == 'L' and side == 'R':
            return child.s
        if parent_assoc == 'R' and side == 'L':
            return child.s
        # otherwise we must wrap
        return f"({child.s})"

    # child is weaker -> must wrap
    return f"({child.s})"

class MathExpressionReconstructor(Transformer[Token, ExprInfo]):
    def number(self, items: list[Token]):
        return ExprInfo(items[0], 5)

    def var(self, items: list[Token]):
        return ExprInfo(items[0], 5)

    def func(self, items: list[Token | ExprInfo]):
        name = items[0]
        if len(items) == 1:     # empty args
            return ExprInfo(f"{name}()", 5)
        args = items[1]
        return ExprInfo(f"{name}({args})", 5)

    def args(self, items: list[ExprInfo]):
        return ",".join(item.s for item in items)

    # ---- Unary ----
    def neg(self, items: list[ExprInfo]):
        child = items[0]
        s = _mathexpression_paren_wrap(child, 3, 'R', 'R')
        return ExprInfo(f"-{s}", 3, 'R')

    # ---- Binary arithmetic ----
    def add(self, items: list[ExprInfo]):
        left, right = items
        left_s = _mathexpression_paren_wrap(left, 1, 'L', 'L')
        right_s = _mathexpression_paren_wrap(right, 1, 'R', 'L')
        return ExprInfo(f"{left_s}+{right_s}", 1, 'L')

    def sub(self, items: list[ExprInfo]):
        left, right = items
        left_s = _mathexpression_paren_wrap(left, 1, 'L', 'L')
        right_s = _mathexpression_paren_wrap(right, 1, 'R', 'L')
        return ExprInfo(f"{left_s}-{right_s}", 1, 'L')

    def mul(self, items: list[ExprInfo]):
        left, right = items
        left_s = _mathexpression_paren_wrap(left, 2, 'L', 'L')
        right_s = _mathexpression_paren_wrap(right, 2, 'R', 'L')
        return ExprInfo(f"{left_s}*{right_s}", 2, 'L')

    def div(self, items: list[ExprInfo]):
        left, right = items
        left_s = _mathexpression_paren_wrap(left, 2, 'L', 'L')
        right_s = _mathexpression_paren_wrap(right, 2, 'R', 'L')
        return ExprInfo(f"{left_s}/{right_s}", 2, 'L')

    # ---- Power ----
    def pow(self, items: list[ExprInfo]):
        left, right = items
        left_s = _mathexpression_paren_wrap(left, 4, 'L', 'R')   # right-associative
        right_s = _mathexpression_paren_wrap(right, 4, 'R', 'R')
        return ExprInfo(f"{left_s}^{right_s}", 4, 'R')

    # ---- Comparisons ----
    def eq(self, _): return "=="
    def ne(self, _): return "!="
    def lt(self, _): return "<"
    def le(self, _): return "<="
    def gt(self, _): return ">"
    def ge(self, _): return ">="

    def compchain1(self, items: list[ExprInfo]):
        left, op, right = items
        s1 = _mathexpression_paren_wrap(left, 0, 'L', 'L')
        s2 = _mathexpression_paren_wrap(right, 0, 'R', 'L')
        return ExprInfo(f"{s1}{op}{s2}", 0, 'L')

    def compchain2(self, items: list[ExprInfo]):
        # a < b < c  →  (no parentheses needed)
        a, op1, b, op2, c = items
        a = _mathexpression_paren_wrap(a, 0, 'L', 'L')
        b = _mathexpression_paren_wrap(b, 0, 'L', 'L')
        c = _mathexpression_paren_wrap(c, 0, 'R', 'L')
        return ExprInfo(f"{a}{op1}{b}{op2}{c}", 0, 'L')

    # Top-level
    def start(self, items: list[ExprInfo]):
        return items[0].s


def get_required_vars(expression: str | Tree[Token]) -> set[str]:
    """Get the set of variable names required by the mathematical expression."""
    if isinstance(expression, str):
        expression = mathexpression_parser.parse(expression) # type: ignore
    
    class VarFinder(Transformer[Token, None]):
        def __init__(self) -> None:
            self.vars: set[str] = set()
        def var(self, n: list[Token]) -> None:
            self.vars.add(n[0].value)
        def evar(self, n: list[Token]) -> None:
            self.vars.add(n[0].value.replace('.', '_'))
    
    vf = VarFinder()
    vf.transform(expression)
    return vf.vars

_T = TypeVar('_T', bound=float | npt.NDArray[Any], covariant=True)
class KwargsCallable(Protocol, Generic[_T]):
    def __call__(self, **kw: float | npt.NDArray[Any]) -> _T: ...

@dataclass(slots=True)
class Expression(Generic[_T]):
    func: KwargsCallable[_T]
    required_vars: set[str]
    source: str
    tree: Tree[Token]

    def __call__(self, **kw: float | npt.NDArray[Any]) -> _T:
        return self.func(**kw)

    @classmethod
    def compile(cls, expression: str) -> Self:
        tree = mathexpression_parser.parse(expression) # type: ignore
        mathcompiler = MathExpressionCompiler()
        pysrc = mathcompiler.transform(tree)

        namespace:dict[str, Any] = {'np': np}
        rqvars = get_required_vars(tree)
        annotation = sorted(rqvars)
        annotation.append('**kw')
        signature = ', '.join(f'{a}: float' for a in annotation)

        fnsrc = f"def func({signature}):\n    return {pysrc}"
        code = compile(fnsrc, '<expr>', 'exec')
        exec(code, namespace)
        return cls(
            required_vars = rqvars,
            func = namespace['func'],
            tree = tree,
            source = expression
        )
    
    def to_string(self) -> str:
        # Reconstruct the expression from the tree, with minimal parentheses and no spaces.
        # This acts as a normalized representation of the expression.
        reconstructor = MathExpressionReconstructor()
        return reconstructor.transform(self.tree).s
