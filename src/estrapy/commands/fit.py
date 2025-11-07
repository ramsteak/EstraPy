import numpy as np

from pathlib import Path

from lark import Token, Tree

from dataclasses import dataclass
from typing import Self

from ..core.grammarclasses import CommandArguments, Command
from ..core.context import Context, ParseContext
from ..grammar.commandparser import CommandArgumentParser
from ..core.number import Number, parse_range, Unit
from ..operations.fourier import apodizer_functions
from ..core.misc import fuzzy_match
from ..core.fitmodel import ExafsModel, load_model_from_yaml



@dataclass(slots=True)
class CommandArguments_Fit(CommandArguments):
    modelpath: Path
    krange: tuple[Number, Number]
    rrange: tuple[Number, Number]
    kweight: float
    apodizer: str
    apodizerp: float

    # to be filled after parsing. If none after parsing, raise error.
    model: ExafsModel = None  # type: ignore

parse_fit_command = CommandArgumentParser(CommandArguments_Fit)
parse_fit_command.add_argument('modelpath', required=True, type=Path)
_default_krange = (Number(None, 0.0, Unit.K), Number(None, np.inf, Unit.K))
_default_rrange = (Number(None, 0.0, Unit.A), Number(None, np.inf, Unit.A))
parse_fit_command.add_argument('krange', '--krange', nargs=2, types=parse_range, default=_default_krange)
parse_fit_command.add_argument('rrange', '--rrange', nargs=2, types=parse_range, default=_default_rrange)
parse_fit_command.add_argument('kweight', '--kweight', '-k', type=float, required=False, default=0.0)
parse_fit_command.add_argument('apodizer', '--apodizer', '-a', type=str, required=False, default='hanning')
parse_fit_command.add_argument('apodizerp', '--parameter', '-p', type=float, required=False, default=3)


@dataclass(slots=True)
class Command_Fit(Command[CommandArguments_Fit]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_fit_command(commandtoken, tokens, parsecontext)

        # Validate parameters
        if arguments.krange[0].unit != Unit.K or arguments.krange[1].unit != Unit.K:
            raise ValueError('Fit krange must be specified in k units.')
        if arguments.krange[0].value < 0.0:
            raise ValueError('Fit krange start must be non-negative.')
        if arguments.krange[1].value <= arguments.krange[0].value:
            raise ValueError('Fit krange end must be greater than start.')
        if arguments.rrange[0].unit != Unit.A or arguments.rrange[1].unit != Unit.A:
            raise ValueError('Fit rrange must be specified in Angstroms.')
        if arguments.rrange[0].value < 0.0:
            raise ValueError('Fit rrange start must be non-negative.')
        if arguments.rrange[1].value <= arguments.rrange[0].value:
            raise ValueError('Fit rrange end must be greater than start.')
        if arguments.kweight < 0.0:
            raise ValueError('Fit kweight must be non-negative.')
        if arguments.apodizer not in apodizer_functions:
            apodizer = fuzzy_match(arguments.apodizer, apodizer_functions)
            if apodizer is not None:
                arguments.apodizer = apodizer
            else:
                raise ValueError(f"Unknown apodizer function '{arguments.apodizer}'. Available options are: {', '.join(apodizer_functions)}.")

        # Resolve and read model
        if not arguments.modelpath.is_absolute():
            arguments.modelpath = (parsecontext.paths.workingdir / arguments.modelpath).resolve()
        
        try:
            _, model = load_model_from_yaml(arguments.modelpath)
            # TODO check version
        except ValueError:
            from ..legacy.model import load_model_from_legacy
            model = load_model_from_legacy(arguments.modelpath)

        arguments.model = model

        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> None:
        pass
