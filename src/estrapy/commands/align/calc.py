import numpy as np

from numpy import typing as npt
from logging import Logger
from scipy.interpolate import make_interp_spline, BSpline
from dataclasses import dataclass
from lark import Token, Tree
from typing import Self, Callable, TypeAlias
from functools import partial

from .shift import SubCommandArguments_Align_Shift, SubCommandResult_Align_Shift

from ...core._validators import validate_number_unit, validate_number_positive, validate_int_non_negative, validate_range_unit
from ...core.context import Command, CommandResult
from ...core.context import Context, ParseContext
from ...core.datastore import Domain, DataPage, ColumnDescription, ColumnKind
from ...core.number import Number, parse_number, parse_range, Unit, parse_edge
from ...core.threaded import execute_threaded
from ...core.commandparser import CommandArgumentParser, CommandArguments, field_arg
from ...operations.edge_detection import correlation_edge_detection, SlidingL2Result


@dataclass(slots=True)
class SubCommandArguments_Align_Calc(CommandArguments):
    energy: Number = field_arg(
        flags = ['--energy', '--E0', '-E'],
        type = parse_edge,
        required = True,
        help = 'Edge energy to align to.',
        validate = validate_number_unit(Unit.EV),
    )
    
    delta: Number = field_arg(
        flags = ['--delta', '-d'],
        type = parse_number,
        required = True,
        help = 'Allowed deviation from the edge energy.',
        validate = [validate_number_unit(Unit.EV), validate_number_positive],
    )

    method: str = field_arg(
        flags = ['--method', '-m'],
        type = str,
        required = False,
        help = 'Method to use for alignment calculation.',
        default = 'set',
    )

    search: Number | None = field_arg(
        flags = ['--search', '--sE0'],
        type = parse_edge,
        required = False,
        help = 'Search energy for the edge if different from the edge energy.',
        default = None,
        validate = validate_number_unit(Unit.EV),
    )
