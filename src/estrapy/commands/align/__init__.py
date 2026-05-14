from dataclasses import dataclass
from lark import Token, Tree
from typing import Self, TypeAlias

from .shift import SubCommandArguments_Align_Shift, SubCommandResult_Align_Shift, execute_shift
from .calc import SubCommandArguments_Align_Calc
from .glitch import SubCommandArguments_Align_Glitch, SubCommandResult_Align_Glitch, execute_glitch

from ...core.context import Command
from ...core.context import Context, ParseContext
from ...core.commandparser import CommandArgumentParser, CommandArguments, field_arg


SubCommandArguments: TypeAlias = SubCommandArguments_Align_Calc | SubCommandArguments_Align_Shift | SubCommandArguments_Align_Glitch

@dataclass(slots=True)
class CommandArguments_Align(CommandArguments):
    mode: SubCommandArguments = field_arg(
        subparsers = {
            'calc': SubCommandArguments_Align_Calc,
            'shift': SubCommandArguments_Align_Shift,
            'glitch': SubCommandArguments_Align_Glitch,
        }
    )

parse_align_command = CommandArgumentParser(CommandArguments_Align, name='align')

CommandResult_Align: TypeAlias = SubCommandResult_Align_Shift | SubCommandResult_Align_Glitch

@dataclass(slots=True)
class Command_Align(Command[CommandArguments_Align, CommandResult_Align]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_align_command.parse(commandtoken, tokens)
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Align:
        match self.args.mode:
            case SubCommandArguments_Align_Calc():
                raise NotImplementedError('Align calc method not implemented yet.')
            
            case SubCommandArguments_Align_Shift():
                return execute_shift(context, self.args.mode)
            case SubCommandArguments_Align_Glitch():
                return execute_glitch(context, self.args.mode)
            case _:
                raise NotImplementedError(f"Unknown mode {self.args.mode} in align command.")
