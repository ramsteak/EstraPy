import pandas as pd
from lark import Token, Tree

from dataclasses import dataclass
from functools import partial
from typing import Self

from ..core.context import Command, CommandResult
from ..core.context import Context, ParseContext
from ..core.commandparser import CommandArgumentParser, CommandArguments, field_arg

from ..core.datastore import Domain, ColumnDescription, ColumnKind
from ..operations.evenodd import diff_even


@dataclass(slots=True)
class CommandArguments_Noise(CommandArguments):
    x: str = field_arg(
        flags=['--xaxiscol'],
        type=str,
        required=False,
        default='E'
    )

    y: str = field_arg(
        flags=['--yaxiscol'],
        type=str,
        required=False,
        default='a'
    )

@dataclass(slots=True)
class CommandResult_Noise(CommandResult):
    ...


parse_noise_command = CommandArgumentParser(CommandArguments_Noise, 'noise')


@dataclass(slots=True)
class Command_Noise(Command[CommandArguments_Noise, CommandResult_Noise]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_noise_command.parse(commandtoken, tokens)

        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Noise:
        column_name = 's' + self.args.y

        expr = partial(estimate_noise, xcol=self.args.x, ycol=self.args.y, name=column_name)

        page0 = next(iter(context.datastore.pages.values()))
        if column_name in page0.domains[Domain.RECIPROCAL].columns:
            raise ValueError(f"Column '{column_name}' already exists in the data. Cannot add noise estimate column.")
        # Get label of y column for use in new column description
        label = "Standard Deviation of " + str(page0.domains[Domain.RECIPROCAL].columns[self.args.y][-1].desc.labl or self.args.y)
        

        for _, page in context.datastore.pages.items():
            datadomain = page.domains[Domain.RECIPROCAL]
            col = ColumnDescription(column_name, None, ColumnKind.DATA, [self.args.x, self.args.y], expr, label, [self.args.y])
            datadomain.add_column(col.name, col)
        
        return CommandResult_Noise()


def estimate_noise(df: pd.DataFrame, xcol: str, ycol: str, name: str) -> pd.Series:
    xy = df[[xcol, ycol]].to_numpy()
    noise = diff_even(xy[:,0], xy[:,1])
    return pd.Series(noise, index=df.index).rename(name)
