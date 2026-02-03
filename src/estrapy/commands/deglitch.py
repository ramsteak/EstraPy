import numpy as np

from numpy import typing as npt
from lark import Token, Tree

from dataclasses import dataclass
from typing import Self

from ..core.context import Context, ParseContext, Command, CommandResult
from ..core.commandparser2 import CommandArgumentParser, CommandArguments, field_arg
from ..core.number import Number, parse_number
from ..operations.derivative import nderivative

@dataclass(slots=True, frozen=True)
class DetectionResult:
    """Result of a glitch detection."""
    evaluation_mask: npt.NDArray[np.bool_]
    glitch_mask: npt.NDArray[np.bool_]
    x: npt.NDArray[np.floating]
    y: npt.NDArray[np.floating]
    std: float | None = None
    baseline: npt.NDArray[np.floating] | None = None

@dataclass(slots=True, frozen=True)
class Finder:
    """Base class for glitch detection methods."""
    def detect(self, x: npt.NDArray[np.floating], y: npt.NDArray[np.floating], i: npt.NDArray[np.bool_]) -> DetectionResult:
        """Detect glitches in the data in the range specified by i."""
        raise NotImplementedError

@dataclass(slots=True, frozen=True)
class Finder_Force(Finder):
    def detect(self, x: npt.NDArray[np.floating], y: npt.NDArray[np.floating], i: npt.NDArray[np.bool_]) -> DetectionResult:
        return DetectionResult(i, i, None, None)

@dataclass(slots=True, frozen=True)
class Finder_Variance(Finder):
    sections: int
    derivative: int

    # To set a pvalue, calculate the inverse threshold from the normal distribution
    threshold: float

    def detect(self, x: npt.NDArray[np.floating], y: npt.NDArray[np.floating], i: npt.NDArray[np.bool_]) -> DetectionResult:
        eval_x = x[i]
        eval_y = y[i]
        if self.derivative:
            eval_y = nderivative(eval_y, self.derivative)
        





        




@dataclass(slots=True)
class CommandArguments_Deglitch(CommandArguments):
    range: tuple[Number, Number]
    

@dataclass(slots=True)
class CommandResult_Deglitch(CommandResult):
    ...

parse_deglitch_command = CommandArgumentParser(CommandArguments_Deglitch)
parse_deglitch_command.add_argument('range', nargs=2, type=parse_number)


@dataclass(slots=True)
class Command_Deglitch(Command[CommandArguments_Deglitch, CommandResult_Deglitch]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_deglitch_command(commandtoken, tokens, parsecontext)
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Deglitch:
        pass
