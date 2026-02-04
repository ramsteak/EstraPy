import numpy as np
from numpy import typing as npt
from logging import Logger

from dataclasses import dataclass
from functools import partial

from .result import BackgroundResult

from ...core.threaded import execute_threaded
from ...core.commandparser import CommandArguments, field_arg
from ...core._validators import validate_float_non_negative, validate_number_unit, validate_number_positive
from ...core.number import Number, parse_number, Unit
from ...operations.fourier import flattop_window, fourier
from ...core.context import Context
from ...core.datastore import Domain


@dataclass(slots=True)
class SubCommand_FourierBackgroundArguments(CommandArguments):
    rmax: Number = field_arg(
        flags=['--rmax'],
        type=parse_number,
        required=False,
        default=Number(None, 1.0, Unit.A),
        validate=[validate_number_unit(Unit.A), validate_number_positive]
    )

    kweight: float = field_arg(
        flags=['--kweight'],
        type=float,
        required=False,
        default=2.0,
        validate=validate_float_non_negative
    )

    forwardpad: Number = field_arg(
        flags=['--forward-pad'],
        type=parse_number,
        required=False,
        default=Number(None, 0.1, Unit.K),
        validate=[validate_number_unit(Unit.K), validate_number_positive]
    )

    forwardwidth: Number = field_arg(
        flags=['--forward-width'],
        type=parse_number,
        required=False,
        default=Number(None, 1.0, Unit.K),
        validate=[validate_number_unit(Unit.K), validate_number_positive]
    )

    backwardpad: Number = field_arg(
        flags=['--backward-pad'],
        type=parse_number,
        required=False,
        default=Number(None, 0.1, Unit.A),
        validate=[validate_number_unit(Unit.A), validate_number_positive]
    )

    backwardwidth: Number = field_arg(
        flags=['--backward-width'],
        type=parse_number,
        required=False,
        default=Number(None, 0.2, Unit.A),
        validate=[validate_number_unit(Unit.A), validate_number_positive]
    )

    epsilon: float = field_arg(
        flags=['--epsilon'],
        type=float,
        required=False,
        default=1e-30,
        validate=validate_float_non_negative
    )


@dataclass(slots=True)
class FourierBackgroundResult(BackgroundResult):
    ...

def _compute_background_fourier(xy: npt.NDArray[np.floating],
                               range: tuple[float, float],
                               sargs: SubCommand_FourierBackgroundArguments,
                               name: str,
                               log: Logger,
                            ) -> FourierBackgroundResult:
    """The function calculates the background using Fourier transform method,
    and returns a result object.
    The arguments are specified by the subcommand arguments, together with the data
    points and range.
    
    All Number instances are considered to be in the correct units."""
    
    # Canonicalize range, converting inf to the finite range
    cutoff = sargs.rmax.value
    kweight = sargs.kweight

    X = xy[:,0]
    minX, maxX = np.min(X), np.max(X)
    range_lower, range_upper = (float(max(range[0], minX)), float(min(range[1], maxX)))

    idx = (xy[:,0] >= range_lower) & (xy[:,0] <= range_upper)

    # Get x and y values within range
    x, y = xy[idx,0], xy[idx,1]

    # Define the transformation space
    r = np.linspace(-5*cutoff, 5*cutoff, 2**10)

    forward_window_shape = (
        range_lower - sargs.forwardpad.value,
        range_lower + sargs.forwardwidth.value,
        range_upper - sargs.forwardwidth.value,
        range_upper + sargs.forwardpad.value
    )
    backward_window_shape = (
        -cutoff - sargs.backwardpad.value,  
        -cutoff + sargs.backwardwidth.value,
        cutoff - sargs.backwardwidth.value,
        cutoff + sargs.backwardpad.value,
    )

    w = flattop_window(x, forward_window_shape, 'hanning')
    W = flattop_window(r, backward_window_shape, 'hanning')

    f = fourier(x, y * x**kweight * w, r)
    b = fourier(r, f.conj() * W, x)

    bkg = b.real / (x**kweight + sargs.epsilon) / w

    Bkg = np.zeros_like(xy[:,1])
    Bkg[idx] = bkg

    log.debug(f'Computed Fourier background for page {name}.')

    return FourierBackgroundResult(
        background = Bkg
    )

def execute_background_fourier(
    context: Context,
    sargs: SubCommand_FourierBackgroundArguments,
    range: tuple[Number, Number],
) -> dict[str, FourierBackgroundResult]:
    
    log = context.logger.getChild('command.background.fourier')
    log.debug(f'Calculating background with Fourier method in range [{range[0]!s}, {range[1]!s}], rmax {sargs.rmax!s}, kweight {sargs.kweight}, window shape: {sargs.forwardpad!s} {sargs.forwardwidth!s}, {sargs.backwardpad!s}, {sargs.backwardwidth!s}.')

    k_range = range[0].value, range[1].value

    log.debug('Preparing data for all pages.')

    page_fulldata: dict[str, npt.NDArray[np.floating]] = {
        name:page.domains[Domain.RECIPROCAL].get_columns_data(['k', 'chi']).to_numpy()
        for name,page in context.datastore.pages.items()
    }

    compute = partial(_compute_background_fourier, range=k_range, sargs=sargs, log=log)

    threaded = len(context.datastore.pages) >= 12 and context.options.debug is False
    page_background = execute_threaded(compute, page_fulldata, argkind='s', threaded=threaded, pass_key_as='name')

    log.info('Completed Fourier background calculation for all pages.')
    
    return page_background
