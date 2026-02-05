import numpy as np
from numpy import typing as npt

from lark import Token, Tree

from dataclasses import dataclass
from typing import Self, TypeAlias, Callable

from ..core.datastore import Domain, ColumnKind, ColumnDescription
from ..core.context import Context, ParseContext, Command, CommandResult
from ..core.commandparser import CommandArgumentParser, CommandArguments, field_arg
from ..core._validators import validate_option_in, type_fuzzy, validate_number_unit
from ..core.misc import infer_axis_domain
from ..core.number import parse_number, Unit, Number


EdgeFunction: TypeAlias = Callable[[npt.NDArray[np.floating],float,float,float,bool], npt.NDArray[np.floating]]

# region Edge Functions

def edge_atan(x: npt.NDArray[np.floating], a: float, b: float, c: float, reparam: bool = True) -> npt.NDArray[np.floating]:
    s = np.tan(0.4 * np.pi) if reparam else 1.0
    return a * (np.atan(s * (x-b)/abs(c)) / np.pi + 0.5)

def edge_tanh(x: npt.NDArray[np.floating], a: float, b: float, c: float, reparam: bool = True) -> npt.NDArray[np.floating]:
    s = np.atanh(0.9) if reparam else 1.0
    return a * (np.tanh(s * (x-b)/abs(c)) + 1) * 0.5

def edge_erf(x: npt.NDArray[np.floating], a: float, b: float, c: float, reparam: bool = True) -> npt.NDArray[np.floating]:
    from scipy.special import erf, erfinv # pyright: ignore[reportMissingTypeStubs]
    s = erfinv(0.8)
    return a * (1 + erf(s * (x-b)/abs(c))) * 0.5

def edge_exp(x: npt.NDArray[np.floating], a: float, b: float, c: float, reparam: bool = True) -> npt.NDArray[np.floating]:
    s = np.log(0.1) if reparam else -1.0

    if c > 0:
        y = np.zeros_like(x)
        i = x > b
        y[i] = a * (1 - np.exp(s * (x[i]-b)/c))
    elif c < 0:
        y = np.full_like(x, a)
        i = x < b
        y[i] = a * np.exp(s * (x[i]-b)/c)
    else:
        y = np.zeros_like(x)
    return y

def edge_linear(x: npt.NDArray[np.floating], a: float, b: float, c: float, reparam: bool = True) -> npt.NDArray[np.floating]:
    if np.isclose(c, 0.0):
        return np.where(x >= b, a, 0.0)
    return a * np.clip((x - b) / abs(c), 0.0, 1.0)

EDGE_FUNCTIONS: dict[str, EdgeFunction] = {
    'atan': edge_atan,
    'tanh': edge_tanh,
    'erf': edge_erf,
    'exp': edge_exp,
    'lin': edge_linear,
}
EDGE_FUNCTION_ALIASES: list[tuple[str, ...]] = [
    ('atan', 'arctangent', 'cauchy', 'lorentzian'),
    ('tanh', 'hyperbolictangent', 'logistic', 'sigmoid'),
    ('erf', 'errorfunction', 'normal', 'gaussian'),
    ('exp', 'exponential', 'onesidedexponential'),
    ('lin', 'linear', 'ramp', 'triangle', 'triangular', 'sawtooth'),
]

# endregion

@dataclass(slots=True)
class CommandArguments_MultiEdge(CommandArguments):
    kind: str = field_arg(
        position=0,
        type=type_fuzzy(EDGE_FUNCTION_ALIASES, min_length=3),
        required=True,
        validate=validate_option_in(EDGE_FUNCTIONS.keys()),
    )
    a: float = field_arg(
        position=1,
        type=float,
        required=True,
    )
    b: Number = field_arg(
        position=2,
        type=parse_number,
        required=True,
        validate=validate_number_unit(Unit.EV, Unit.K)
    )
    c: Number = field_arg(
        position=3,
        type=parse_number,
        required=True,
        validate=validate_number_unit(Unit.EV, Unit.K)
    )
    axis: str = field_arg(
        flags=['--axis', '-x'],
        type=str,
        required=False,
        default=None,
    )
    reparam: bool = field_arg(
        flags=['--reparam', '-r'],
        nargs=0,
        action='store_false',
        required=False,
        default=True,
    )
    column: str = field_arg(
        flags=['--column', '-c'],
        type=str,
        required=False,
        default='a',
    )

@dataclass(slots=True)
class CommandResult_MultiEdge(CommandResult):
    ...

parse_multiedge_command = CommandArgumentParser(CommandArguments_MultiEdge, 'multiedge')


@dataclass(slots=True)
class Command_MultiEdge(Command[CommandArguments_MultiEdge, CommandResult_MultiEdge]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_multiedge_command.parse(commandtoken, tokens)
        
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_MultiEdge:
        axis, _ = infer_axis_domain(axis = self.args.axis, numbers=[self.args.b, self.args.c], domain=Domain.RECIPROCAL)
        edge_func = EDGE_FUNCTIONS[self.args.kind]

        for _, page in context.datastore.pages.items():
            domain = page.domains[Domain.RECIPROCAL]
            x = domain.get_column_data(axis).to_numpy()
            y = edge_func(x, self.args.a, self.args.b.value, self.args.c.value, self.args.reparam)

            col_name = f'edge_{self.args.kind}'
            col_desc = ColumnDescription(
                name = col_name,
                type = ColumnKind.DATA,
                unit = None,
                labl = f'Multiple edge compensation ({self.args.kind})',
            )
            domain.add_column_data(col_name, col_desc, y)

            new = ColumnDescription(
                name=self.args.column,
                unit=None,
                type=ColumnKind.DATA,
                deps=[self.args.column, col_name],
                calc=lambda d,c1=self.args.column,c2=col_name: (d[c1] - d[c2])
            )
            domain.add_column(self.args.column, new)



        return CommandResult_MultiEdge()
