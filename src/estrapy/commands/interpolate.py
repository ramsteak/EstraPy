import numpy as np
import pandas as pd

from scipy.interpolate import make_interp_spline # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from lark import Token, Tree
from dataclasses import dataclass
from typing import Self

from ..core.context import CommandArguments, Command, CommandResult
from ..core.context import Context, ParseContext
from ..core.commandparser import CommandArgumentParser
from ..core.number import Number, parse_number, Unit, parse_range
from ..core.datastore import Domain, ColumnKind
from ..core.misc import fuzzy_match

@dataclass(slots=True)
class CommandArguments_Interpolate(CommandArguments):
    range: tuple[Number, Number]
    interval: Number | None
    number: int | None
    axis: str
    domain: Domain

@dataclass(slots=True)
class CommandResult_Interpolate(CommandResult):
    ...

parse_interpolate_command = CommandArgumentParser(CommandArguments_Interpolate)
parse_interpolate_command.add_argument('range', types=parse_range, nargs=2, required=True, default=None)
parse_interpolate_command.add_argument('interval', '--interval', type=parse_number, required=False, default=None)
parse_interpolate_command.add_argument('number', '--number', type=int, required=False, default=None)
parse_interpolate_command.add_argument('axis', '--axis', type=str, required=False, default=None)
parse_interpolate_command.add_argument('domain', '--domain', type=str, required=False, default=None)

@dataclass(slots=True)
class Command_Interpolate(Command[CommandArguments_Interpolate, CommandResult_Interpolate]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_interpolate_command(commandtoken, tokens, parsecontext)
        log = parsecontext.logger.getChild(f'parse.interpolate')

        # Require one of interval or number to be set
        if arguments.interval is None and arguments.number is None: # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError("Interpolate command requires either --interval or --number to be specified.")
        if arguments.interval is not None and arguments.number is not None: # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError("Interpolate command requires only one of --interval or --number to be specified.")
        
        # TODO: whenever, add other domain support
        if arguments.domain is not None: # pyright: ignore[reportUnnecessaryComparison]
            domain = fuzzy_match(arguments.domain, ['reciprocal', 'fourier'])  # pyright: ignore[reportArgumentType]
        else:
            domain = None

        if arguments.axis is None: # pyright: ignore[reportUnnecessaryComparison]
            # Try to infer axis from range unit and sign
            arguments.axis = 'E'  # Default axis if not specified

            match arguments.range, arguments.interval, domain:
                case (Number(unit=None), Number(unit=None)), Number(unit=None), None | 'reciprocal': # pyright: ignore[reportUnnecessaryComparison]
                    log.warning("Cannot infer axis from range with no units. Defaulting to 'E'.")
                    arguments.axis = 'E'
                case (Number(unit=Unit.EV | None, sign=None), Number(unit=Unit.EV | None, sign=None)), Number(unit=Unit.EV | None), None | 'reciprocal': # pyright: ignore[reportUnnecessaryComparison]
                    arguments.axis = 'E'
                case (Number(unit=Unit.EV | None, sign='+'|'-'), Number(unit=Unit.EV | None, sign='+'|'-')), Number(unit=Unit.EV | None), None | 'reciprocal': # pyright: ignore[reportUnnecessaryComparison]
                    arguments.axis = 'e'
                case (Number(unit=Unit.K | None), Number(unit=Unit.K | None)), Number(unit=Unit.K | None), None | 'reciprocal': # pyright: ignore[reportUnnecessaryComparison]
                    arguments.axis = 'k'
                case (Number(unit=Unit.A | None), Number(unit=Unit.A | None)), Number(unit=Unit.A | None), None | 'fourier': # pyright: ignore[reportUnnecessaryComparison]
                    arguments.axis = 'R'
                case _:
                    raise ValueError("Cannot infer axis from range units. Specify the --axis argument.")
        
        arguments.domain = Domain(domain)

        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Interpolate:
        log = context.logger.getChild(f'command.interpolate')
        range = self.args.range[0].value, self.args.range[1].value
        if self.args.interval is not None:
            interval = self.args.interval.value
            new_axis = np.arange(range[0], range[1] + interval, interval)
        elif self.args.number is not None:
            number = self.args.number
            new_axis = np.array(np.linspace(range[0], range[1], number)) # pyright: ignore[reportCallIssue, reportUnknownArgumentType]
        else:
            raise ValueError("Interpolate command requires either --interval or --number to be specified. Also, this should never have happened.")
        
        axisname = self.args.axis

        for page in context.datastore.pages.values():
            domain = page.domains[self.args.domain]
            if axisname not in domain.columns:
                raise ValueError(f"Axis '{axisname}' not found in domain '{self.args.domain.value}' for page '{page.meta.name}'.")
            if domain.columns[axisname][-1].desc.type != ColumnKind.AXIS:
                raise ValueError(f"Column '{axisname}' in domain '{self.args.domain.value}' for page '{page.meta.name}' is not an axis column.")
            
            old_axis = domain.get_column_data(axisname).to_numpy()
            # TODO: handle this transformation in a non destructive way
            
            new_data = pd.DataFrame(
                make_interp_spline(old_axis, domain.data.to_numpy(), k=3)(new_axis), # pyright: ignore[reportUnknownArgumentType]
                columns=domain.data.columns,
            )
            domain.data = new_data
            log.debug(f"Interpolated page '{page.meta.name}' in domain '{self.args.domain.value}' on axis '{axisname}' to new axis with {len(new_axis)} points.")
        log.info(f"Interpolated all pages in domain '{self.args.domain.value}' on axis '{axisname}'.")

        return CommandResult_Interpolate()
