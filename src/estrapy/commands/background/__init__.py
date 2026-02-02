import numpy as np

from lark import Token, Tree

from dataclasses import dataclass
from typing import Self

from .result import CommandResult_Background
from .fourier_background import execute_background_fourier, SubCommand_FourierBackgroundArguments, subcommand_fourier
from .spline_background import execute_background_spline, SubCommand_SplineBackgroundArguments, subcommand_spline
from .polynomial_background import execute_background_polynomial, SubCommand_PolynomialBackgroundArguments, subcommand_polynomial

from ...core.context import CommandArguments, Command, Context, ParseContext
from ...core.commandparser import CommandArgumentParser
from ...core.number import Number, parse_range, Unit
from ...core.datastore import Domain, ColumnDescription, ColumnKind


@dataclass(slots=True)
class CommandArguments_Background(CommandArguments):
    range: tuple[Number, Number]
    mode: SubCommand_FourierBackgroundArguments | SubCommand_SplineBackgroundArguments | SubCommand_PolynomialBackgroundArguments

_default_range = (Number(None, 0.0, Unit.K), Number(None, np.inf, Unit.K))
parse_background_command = CommandArgumentParser(CommandArguments_Background)
parse_background_command.add_argument('range', types=parse_range, nargs=2, required=False, default=_default_range)
parse_background_command.add_subparser('fourier', subcommand_fourier, 'mode')
parse_background_command.add_subparser('spline', subcommand_spline, 'mode')
parse_background_command.add_subparser('polynomial', subcommand_polynomial, 'mode')


@dataclass(slots=True)
class Command_Background(Command[CommandArguments_Background, CommandResult_Background]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_background_command(commandtoken, tokens, parsecontext)

        # Allow infinite bounds without an explicit unit (e.g. "0k ..") — treat bounds with unit None as valid only if they are infinite
        def _is_k_or_infinite(num: Number) -> bool:
            try:
                return (num.unit is Unit.K) or (num.unit is None and np.isinf(num.value))
            except Exception:
                return False

        if not (_is_k_or_infinite(arguments.range[0]) and _is_k_or_infinite(arguments.range[1])):
            raise ValueError('Background range must be specified in k units.')
        
        # Validate subcommand arguments
        match arguments.mode:
            # TODO: when the new validation framework is in place, move these checks there
            case SubCommand_FourierBackgroundArguments(rmax=rmax):
                if rmax.unit is not Unit.A:
                    raise ValueError('Background Fourier rmax must be specified in Angstroms.')
            case SubCommand_PolynomialBackgroundArguments(degree=degree):
                if degree < 0:
                    raise ValueError('Background Polynomial degree must be non-negative.')
            case SubCommand_SplineBackgroundArguments():
                ...
                # if nodes is not None and nknots is not None:
                #     raise ValueError('Background Spline: Specify either nodes or nknots, not both.')
                # elif nodes is None and nknots is None:
                #     raise ValueError('Background Spline: Must specify either nodes or nknots.')
                
                # if nknots is not None and nknots <= 0:
                #     raise ValueError('Background Spline nknots must be positive.')
                
                # if nodes is not None:
                #     for node in nodes:
                #         if node.unit is not Unit.K:
                #             raise ValueError('Background Spline nodes must be specified in k units.')
                
            case _:
                pass

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