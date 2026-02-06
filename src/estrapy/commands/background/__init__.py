import numpy as np

from lark import Token, Tree

from dataclasses import dataclass
from typing import Self

from .result import CommandResult_Background
from .fourier_background import execute_background_fourier, SubCommand_FourierBackgroundArguments
from .spline_background import execute_background_spline, SubCommand_SplineBackgroundArguments
from .polynomial_background import execute_background_polynomial, SubCommand_PolynomialBackgroundArguments

from ...core.context import Command, Context, ParseContext
from ...core.commandparser import CommandArgumentParser, CommandArguments, field_arg
from ...core._validators import validate_range_unit
from ...core.number import Number, parse_range, Unit
from ...core.datastore import Domain, ColumnDescription, ColumnKind


@dataclass(slots=True)
class CommandArguments_Background(CommandArguments):
    mode: (
        SubCommand_FourierBackgroundArguments | 
        SubCommand_SplineBackgroundArguments | 
        SubCommand_PolynomialBackgroundArguments
    ) = field_arg(
        subparsers={
            'fourier': SubCommand_FourierBackgroundArguments,
            'spline': SubCommand_SplineBackgroundArguments,
            'polynomial': SubCommand_PolynomialBackgroundArguments,
        }
    )

    
    range: tuple[Number, Number] = field_arg(
        types=parse_range,
        nargs=2,
        required=False,
        default=(Number(None, 0.0, Unit.K), Number(None, np.inf, Unit.K)),
        validate=validate_range_unit(Unit.K, Unit.EV)
    )


parse_background_command = CommandArgumentParser(CommandArguments_Background, 'background')


@dataclass(slots=True)
class Command_Background(Command[CommandArguments_Background, CommandResult_Background]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_background_command.parse(commandtoken, tokens)

        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Background:
        match self.args.mode:
            case SubCommand_FourierBackgroundArguments() as sargs:
                backgrounds = execute_background_fourier(context, sargs, self.args.range)
            case SubCommand_SplineBackgroundArguments() as sargs:
                backgrounds = execute_background_spline(context, sargs, self.args.range)
            case SubCommand_PolynomialBackgroundArguments() as sargs:
                backgrounds = execute_background_polynomial(context, sargs, self.args.range)
            case _:
                raise ValueError('Unknown background mode.')
        # Backgrounds is a dict of page name -> BackgroundResult

        for name, page in context.datastore.pages.items():
            domain = page.domains[Domain.RECIPROCAL]
            bkg = backgrounds[name].background
            bkgcol = ColumnDescription(name='bkg', type=ColumnKind.DATA, unit=None, labl='Background')
            domain.add_column_data('bkg', bkgcol, bkg)
            chicol = ColumnDescription(name='chi', type=ColumnKind.DATA, unit=None, deps=['bkg', 'chi'], calc=lambda df: df['chi'] - df['bkg'], labl='EXAFS signal')
            domain.add_column('chi', chicol)
        
        return CommandResult_Background(
        )