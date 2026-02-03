import numpy as np
from numpy import typing as npt
from logging import Logger

from dataclasses import dataclass
from functools import partial

from .result import BackgroundResult

from ...core.threaded import execute_threaded
from ...core.commandparser2 import CommandArguments, field_arg
from ...core._validators import validate_int_non_negative, validate_float_non_negative
from ...core.number import Number
from ...core.context import Context
from ...core.datastore import Domain

@dataclass(slots=True)
class SubCommand_PolynomialBackgroundArguments(CommandArguments):
    degree: int = field_arg(
        flags=['--degree', '-d'],
        type=int,
        required=False,
        default=3,
        const_flags={
            '--linear': 1, '-l': 1,
            '--quadratic': 2, '-q': 2,
            '--cubic': 3, '-c': 3
        },
        validate=validate_int_non_negative
    )

    kweight: float = field_arg(
        flags=['--kweight'],
        type=float,
        required=False,
        default=2.0,
        validate=validate_float_non_negative
    )


@dataclass(slots=True)
class PolynomialBackgroundResult(BackgroundResult):
    ...

def _compute_background_polynomial(xy: npt.NDArray[np.floating],
                               range: tuple[float, float],
                               sargs: SubCommand_PolynomialBackgroundArguments,
                               name: str,
                               log: Logger,
                            ) -> PolynomialBackgroundResult:
    """The function calculates the background using a polynomial regression method,
    and returns a result object.
    The arguments are specified by the subcommand arguments, together with the data
    points and range.
    
    All Number instances are considered to be in the correct units."""
    
    # Canonicalize range, converting inf to the finite range
    degree = sargs.degree
    kweight = sargs.kweight

    X = xy[:,0]
    minX, maxX = np.min(X), np.max(X)
    range_lower, range_upper = (float(max(range[0], minX)), float(min(range[1], maxX)))

    idx = (xy[:,0] >= range_lower) & (xy[:,0] <= range_upper)

    # Get x and y values within range
    x, y = xy[idx,0], xy[idx,1]

    # Fit polynomial
    poly = np.poly1d(np.polyfit(x, y * x**kweight, degree))
    bkg = poly(X) / (X ** kweight)

    log.debug(f'Computed polynomial background for page {name}.')

    return PolynomialBackgroundResult(
        background = bkg,
    )

def execute_background_polynomial(
    context: Context,
    sargs: SubCommand_PolynomialBackgroundArguments,
    range: tuple[Number, Number],
) -> dict[str, PolynomialBackgroundResult]:
    
    log = context.logger.getChild('command.background.polynomial')
    log.debug(f'Calculating background with Polynomial method in range [{range[0]!s}, {range[1]!s}], degree {sargs.degree}, kweight {sargs.kweight}.')

    k_range = range[0].value, range[1].value

    page_fulldata: dict[str, npt.NDArray[np.floating]] = {
        name:page.domains[Domain.RECIPROCAL].get_columns_data(['k', 'chi']).to_numpy()
        for name,page in context.datastore.pages.items()
    }

    log.debug('Computing background for all pages.')
    compute = partial(_compute_background_polynomial, range=k_range, sargs=sargs, log=log)

    threaded = len(context.datastore.pages) >= 12 and context.options.debug is False
    page_background = execute_threaded(compute, page_fulldata, argkind='s', threaded=threaded, pass_key_as='name')

    return page_background
