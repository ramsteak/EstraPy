from lark import Token, Tree

from dataclasses import dataclass
from typing import Self

from ..core.context import Command, CommandResult
from ..core.context import Context, ParseContext
from ..core.commandparser import CommandArgumentParser, CommandArguments, field_arg
from ..core.number import Number, parse_range
from ..core.datastore import Domain

@dataclass(slots=True)
class CommandArguments_Cut(CommandArguments):
    range: tuple[Number, Number] = field_arg(
        position=0,
        types=parse_range,
        nargs=2,
        required=True,
    )

@dataclass(slots=True)
class CommandResult_Cut(CommandResult):
    ...

parse_cut_command = CommandArgumentParser(CommandArguments_Cut, 'cut')


@dataclass(slots=True)
class Command_Cut(Command[CommandArguments_Cut, CommandResult_Cut]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_cut_command.parse(commandtoken, tokens)
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Cut:
        log = context.logger.getChild('command.cut')

        # TODO assume always absolute energy for now
        start, end = self.args.range[0].value, self.args.range[1].value

        for name, page in context.datastore.pages.items():
            domain = page.domains[Domain.RECIPROCAL]

            energycol = domain.get_column_data("E")
            mask = (energycol >= start) & (energycol <= end)
            # TODO: this is very destructive, should make a copy and store an index or something
            domain.data = domain.data[mask]
        
            log.debug(f"Cut command applied to page '{name}': kept data in range {start} to {end} eV")
        
        log.info(f"Cut command executed: kept data in range {start} to {end} eV for all pages")
        
        return CommandResult_Cut()