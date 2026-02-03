import numpy as np
import pandas as pd

from scipy.interpolate import make_interp_spline # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from lark import Token, Tree
from dataclasses import dataclass
from typing import Self

from ..core.context import Command, CommandResult
from ..core.context import Context, ParseContext
from ..core.commandparser2 import CommandArgumentParser, CommandArguments, field_arg
from ..core._validators import validate_number_positive, validate_int_positive, type_enum, validate_non_null
from ..core.number import Number, parse_number, Unit, parse_range
from ..core.datastore import Domain, ColumnKind

@dataclass(slots=True)
class CommandArguments_Interpolate(CommandArguments):
    range: tuple[Number, Number] = field_arg(
        position=0,
        types=parse_range,
        nargs=2,
        required=True,
        default=None,
    )

    interval: Number | None = field_arg(
        flags=['--interval'],
        type=parse_number,
        required=False,
        default=None,
        validate=validate_number_positive
    )

    number: int | None = field_arg(
        flags=['--number'],
        type=int,
        required=False,
        default=None,
        validate=validate_int_positive
    )

    axis: str | None = field_arg(
        flags=['--axis'],
        type=str,
        required=False,
        default=None
    )

    domain: Domain = field_arg(
        flags=['--domain'],
        type=type_enum(Domain),
        required=False,
        default=None,
        validate=validate_non_null
    )

    def validate(self) -> None:
        if self.interval is None and self.number is None:
            raise ValueError("Interpolate command requires either --interval or --number to be specified.")
        if self.interval is not None and self.number is not None:
            raise ValueError("Interpolate command requires only one of --interval or --number to be specified.")
    
    def __post_init__(self) -> None:
        # Try to infer axis from range unit and sign
        if self.axis is None: # pyright: ignore[reportUnnecessaryComparison]
            # Try to infer axis from range unit and sign

            match self.range, self.interval, self.domain:
                case (Number(unit=None), Number(unit=None)), Number(unit=None), None | Domain.RECIPROCAL:
                    self.axis = 'E'
                    self.domain = Domain.RECIPROCAL
                case (Number(unit=Unit.EV | None, sign=None), Number(unit=Unit.EV | None, sign=None)), Number(unit=Unit.EV | None), None | Domain.RECIPROCAL:
                    self.axis = 'E'
                    self.domain = Domain.RECIPROCAL
                case (Number(unit=Unit.EV | None, sign='+'|'-'), Number(unit=Unit.EV | None, sign='+'|'-')), Number(unit=Unit.EV | None), None | Domain.RECIPROCAL:
                    self.axis = 'e'
                    self.domain = Domain.RECIPROCAL
                case (Number(unit=Unit.K | None), Number(unit=Unit.K | None)), Number(unit=Unit.K | None), None | Domain.RECIPROCAL:
                    self.axis = 'k'
                    self.domain = Domain.RECIPROCAL
                case (Number(unit=Unit.A | None), Number(unit=Unit.A | None)), Number(unit=Unit.A | None), None | Domain.FOURIER:
                    self.axis = 'R'
                    self.domain = Domain.FOURIER
                case _:
                    raise ValueError("Cannot infer axis from range units. Specify the --axis argument.")
        

@dataclass(slots=True)
class CommandResult_Interpolate(CommandResult):
    ...

parse_interpolate_command = CommandArgumentParser(CommandArguments_Interpolate, 'interpolate')

@dataclass(slots=True)
class Command_Interpolate(Command[CommandArguments_Interpolate, CommandResult_Interpolate]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_interpolate_command.parse(commandtoken, tokens)

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
