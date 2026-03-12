from lark import Token, Tree

from dataclasses import dataclass
from typing import Self

from ..core.context import Command, CommandResult
from ..core.context import Context, ParseContext
from ..core.commandparser import CommandArgumentParser, CommandArguments, field_arg
from ..core._validators import validate_number_positive, validate_int_positive, type_enum, validate_option_in
from ..core.number import Number, parse_number, parse_range
from ..core.datastore import Domain, ColumnKind
from ..core.misc import infer_axis_domain

@dataclass(slots=True)
class SubCommandArguments_EnoughAxis(CommandArguments):
    range: tuple[Number, Number] = field_arg(
        types=parse_range,
        nargs=2,
        required=True,
        default=None,
    )

    axis: str = field_arg(
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
        validate=validate_option_in(Domain)
    )

    def __post_init__(self) -> None:
        # Try to infer axis from range unit and sign
        axis, domain = infer_axis_domain(axis=self.axis, range=self.range, domain=self.domain)
        self.axis = axis
        self.domain = domain
        


@dataclass(slots=True)
class CommandArguments_Filter(CommandArguments):
    mode: SubCommandArguments_EnoughAxis = field_arg(
        subparsers={
            'enoughaxis': SubCommandArguments_EnoughAxis,
        }
    )

@dataclass(slots=True)
class CommandResult_Filter(CommandResult):
    ...

parse_filter_command = CommandArgumentParser(CommandArguments_Filter, 'filter')


@dataclass(slots=True)
class Command_Filter(Command[CommandArguments_Filter, CommandResult_Filter]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_filter_command.parse(commandtoken, tokens)
        
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Filter:
        
        match self.args.mode:
            case SubCommandArguments_EnoughAxis():
                log = context.logger.getChild('filter.enoughaxis')

                range_start, range_end = self.args.mode.range[0].value, self.args.mode.range[1].value

                to_remove: list[str] = []
                for name, page in context.datastore.pages.items():
                    domain = page.domains[self.args.mode.domain]
                    axis = domain.get_column_data(self.args.mode.axis)

                    # Filter files that do not have enough axis values in the specified range
                    axmin, axmax = axis.min(), axis.max()
                    if axmin > range_start or axmax < range_end:
                        log.warning(f"Filtering out page '{name}' because axis '{self.args.mode.axis}' ({axmin}, {axmax}) does not cover the range [{range_start}, {range_end}].")
                        to_remove.append(name)
                
                for name in to_remove:
                    context.datastore.pages.pop(name, None)
                
                log.info(f"Filtered out {len(to_remove)} pages that do not have enough axis values in the specified range [{range_start}, {range_end}].")
                return CommandResult_Filter()