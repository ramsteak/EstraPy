import numpy as np
import pandas as pd

from lark import Token, Tree

from dataclasses import dataclass
from typing import Self

from scipy.interpolate import make_interp_spline # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from ..core.datastore import Domain
from ..core.context import Context, ParseContext, Command, CommandResult
from ..core.commandparser import CommandArgumentParser, CommandArguments, field_arg
from ..core._validators import validate_option_in, type_enum, validate_number_positive, validate_noninfinite_range, type_bool
from ..core.number import Number, parse_range, parse_number
from ..core.misc import infer_axis_domain

@dataclass(slots=True)
class CommandArguments_Rebin(CommandArguments):
    range: tuple[Number, Number] = field_arg(
        types=parse_range,
        nargs=2,
        required=True,
        default=None,
        validate=validate_noninfinite_range
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
        validate=validate_number_positive
    )
    
    axis: str = field_arg(
        flags=['--axis', '-x'],
        type=str,
        required=False,
        default=None,
    )
    
    domain: Domain = field_arg(
        flags=['--domain'],
        type=type_enum(Domain),
        required=False,
        default=Domain.RECIPROCAL,
        validate=validate_option_in(Domain),
    )

    fix_points: bool = field_arg(
        flags=['--fix-points'],
        type=bool,
        action='store_true',
        nargs=0,
        required=False,
        default=False,
        help="Whether to ensure that the rebinned points are exactly as specified. If not set, the rebinned points will be the mean of the points in each bin."
    )

    def validate(self) -> None:
        if self.interval is None and self.number is None:
            raise ValueError("Rebin command requires either --interval or --number to be specified.")
        if self.interval is not None and self.number is not None:
            raise ValueError("Rebin command requires only one of --interval or --number to be specified.")
    
    def __post_init__(self) -> None:
        axis, domain = infer_axis_domain(domain = self.domain, axis = self.axis, numbers = [self.interval], range = self.range)
        self.axis = axis
        self.domain = domain

@dataclass(slots=True)
class CommandResult_Rebin(CommandResult):
    ...

parse_rebin_command = CommandArgumentParser(CommandArguments_Rebin, 'rebin')


@dataclass(slots=True)
class Command_Rebin(Command[CommandArguments_Rebin, CommandResult_Rebin]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_rebin_command.parse(commandtoken, tokens)
        
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Rebin:
        log = context.logger.getChild(f'command.rebin')
        
        if self.args.interval is not None:
            interval = self.args.interval.value
            # Boundaries define the edges of the bins. There is one more boundary,
            # and they are shifted by half an interval compared to the bin centers.
            range_low, range_high = self.args.range[0].value, self.args.range[1].value
            interval = self.args.interval.value
            new_axis = np.arange(range_low, range_high + interval, interval)
            boundaries = np.arange(range_low - 0.5 * interval, range_high + interval, interval)

        elif self.args.number is not None:
            number = self.args.number
            range_low, range_high = self.args.range[0].value, self.args.range[1].value
            new_axis = np.linspace(range_low, range_high, number)
            boundaries = np.array(np.linspace(range_low - 0.5, range_high + 0.5, number + 1), dtype=float)
        else:
            raise ValueError("Interpolate command requires either --interval or --number to be specified. Also, this should never have happened.")
        
        for page in context.datastore.pages.values():
            axis_column_name = page.domains[self.args.domain]._resolve_columns([self.args.axis])[0]
            axis = page.domains[self.args.domain].get_column_data(self.args.axis).to_numpy()
            
            # Careful that the range might extend beyond the data limits, so we need
            # to remove -1 values and len(boundaries)
            bins = np.digitize(axis, boundaries) - 1

            rebinned = page.domains[self.args.domain].data.groupby(bins).mean().iloc[1:-1].reset_index(drop=True) # pyright: ignore[reportUnknownMemberType]

            if self.args.fix_points:
                new_data = np.asarray(
                        make_interp_spline(
                            rebinned[axis_column_name].to_numpy(),
                            rebinned.to_numpy(),
                            3)(new_axis), # pyright: ignore[reportUnknownArgumentType]
                )

                rebinned = pd.DataFrame(new_data, columns=rebinned.columns)
            
            rebinned[axis_column_name] = new_axis
            page.domains[self.args.domain].data = rebinned

            log.debug(f"Rebinned page '{page.meta.name}' in domain '{self.args.domain.value}' on axis '{self.args.axis}' to new axis with {len(boundaries) - 1} points.")
        log.info(f"Rebinned all pages in domain '{self.args.domain.value}' on axis '{self.args.axis}'. Note that this operation is destructive.")
    
        return CommandResult_Rebin()
    