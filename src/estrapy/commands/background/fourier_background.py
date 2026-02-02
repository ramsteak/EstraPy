import numpy as np
from numpy import typing as npt
from logging import Logger

from dataclasses import dataclass
from functools import partial

from .result import BackgroundResult

from ...core.threaded import execute_threaded
from ...core.commandparser import CommandArgumentParser
from ...core.context import CommandArguments
from ...core.number import Number, parse_number
from ...operations.fourier import flattop_window, fourier
from ...core.context import Context
from ...core.datastore import Domain


@dataclass(slots=True)
class SubCommand_FourierBackgroundArguments(CommandArguments):
    rmax: Number
    kweight: float
    forwardpad: Number = parse_number('0.1k')
    forwardwidth: Number = parse_number('1.0k')
    backwardpad: Number = parse_number('0.1A')
    backwardwidth: Number = parse_number('0.2A')

subcommand_fourier = CommandArgumentParser(
    SubCommand_FourierBackgroundArguments,
    name = 'fourier'
)

subcommand_fourier.add_argument('rmax', '--rmax', type = parse_number, required=False, default=parse_number('1A'))
subcommand_fourier.add_argument('kweight', '--kweight', type = float, required=False, default=2.0)
subcommand_fourier.add_argument('forwardpad', '--forward-pad', type = parse_number, required=False, default=parse_number('0.1k'))
subcommand_fourier.add_argument('forwardwidth', '--forward-width', type = parse_number, required=False, default=parse_number('1.0k'))
subcommand_fourier.add_argument('backwardpad', '--backward-pad', type = parse_number, required=False, default=parse_number('0.1A'))
subcommand_fourier.add_argument('backwardwidth', '--backward-width', type = parse_number, required=False, default=parse_number('0.2A'))

@dataclass(slots=True)
class FourierBackgroundResult(BackgroundResult):
    ...

FORWARD_WINDOW_PADDING = 0.1
FORWARD_WINDOW_WIDTH = 1.0

BACKWARD_WINDOW_PADDING = 0.1
BACKWARD_WINDOW_WIDTH = 0.2

EPSILON = 1e-30
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
        range_lower - FORWARD_WINDOW_PADDING,
        range_lower + FORWARD_WINDOW_WIDTH,
        range_upper - FORWARD_WINDOW_WIDTH,
        range_upper + FORWARD_WINDOW_PADDING
    )
    backward_window_shape = (
        -cutoff - BACKWARD_WINDOW_PADDING,  
        -cutoff + BACKWARD_WINDOW_WIDTH,
        cutoff - BACKWARD_WINDOW_WIDTH,
        cutoff + BACKWARD_WINDOW_PADDING
    )

    w = flattop_window(x, forward_window_shape, 'hanning')
    W = flattop_window(r, backward_window_shape, 'hanning')

    f = fourier(x, y * x**kweight * w, r)
    b = fourier(r, f.conj() * W, x)

    bkg = b.real / (x**kweight + EPSILON) / w

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

    log.debug('Computing background for all pages.')
    compute = partial(_compute_background_fourier, range=k_range, sargs=sargs, log=log)

    threaded = len(context.datastore.pages) >= 12 and context.options.debug is False
    page_background = execute_threaded(compute, page_fulldata, argkind='s', threaded=threaded, pass_key_as='name')

    return page_background
