import numpy as np
from numpy import typing as npt
from logging import Logger

from dataclasses import dataclass
from functools import partial

from .result import BackgroundResult

from ...core.threaded import execute_threaded
from ...core.commandparser import CommandArguments, field_arg
from ...core._validators import validate_float_non_negative, validate_int_non_negative
from ...core.number import Number, parse_number, Unit
from ...core.context import Context
from ...core.datastore import Domain

from ...operations.autobk import AutobkFitter

@dataclass(slots=True)
class SubCommand_AutobkBackgroundArguments(CommandArguments):
    rbkg: Number = field_arg(
        flags=['--rbkg'],
        type=parse_number,
        required=False,
        default=Number(None, 1.0, Unit.A)
    )

    kweight: float = field_arg(
        flags=['--kweight'],
        type=float,
        required=False,
        default=2.0,
        validate=validate_float_non_negative
    )

    nknots: int | None = field_arg(
        flags=['--nknots'],
        type=int,
        required=False,
        default=None,
        validate=validate_int_non_negative
    )

    def validate(self) -> bool:
        """
        Validates the arguments specific to the AUTOBK algorithm.
        """
        if self.rbkg.value <= 0:
            raise ValueError(f"The R-background cutoff (rbkg) must be strictly positive. Got {self.rbkg.value}.")
        return True


@dataclass(slots=True)
class AutobkBackgroundResult(BackgroundResult):
    ...


def _compute_background_autobk(xy: npt.NDArray[np.floating],
                               range: tuple[float, float],
                               sargs: SubCommand_AutobkBackgroundArguments,
                               autobk_fitter: AutobkFitter,
                               name: str,
                               log: Logger,
                            ) -> AutobkBackgroundResult:
    """The function calculates the background using the AUTOBK method,
    and returns a result object.
    The arguments are specified by the subcommand arguments, together with the data
    points and range.
    
    All Number instances are considered to be in the correct units."""
    
    kweight = sargs.kweight

    X = xy[:,0]
    minX, maxX = np.min(X), np.max(X)
    range_lower, range_upper = (float(max(range[0], minX)), float(min(range[1], maxX)))

    idx = (xy[:,0] >= range_lower) & (xy[:,0] <= range_upper)

    # Get x and y values within range
    x, y = xy[idx,0], xy[idx,1]

    # Calculate background (assuming fitter handles the optimization loop)
    # y is weighted before passing to the fitter, and un-weighted afterwards
    bkg = autobk_fitter.fit(x, y * x**kweight) / (x ** kweight)
    
    Bkg = np.zeros_like(xy[:,1])
    Bkg[idx] = bkg

    log.debug(f'Computed AUTOBK background for page {name}.')

    return AutobkBackgroundResult(
        background = Bkg
    )


def execute_background_autobk(
    context: Context,
    sargs: SubCommand_AutobkBackgroundArguments,
    range: tuple[Number, Number],
) -> dict[str, AutobkBackgroundResult]:
    
    log = context.logger.getChild('command.background.autobk')
    log.debug(
        f'Executing AUTOBK background subtraction in range [{range[0]}, {range[1]}], '
        f'kweight={sargs.kweight}, rbkg={sargs.rbkg}, nknots={sargs.nknots}'
    )

    k_range = range[0].value, range[1].value
    delta_k = k_range[1] - k_range[0]

    # Nyquist limit for degrees of freedom: N = (2 * delta_k * Rbkg) / pi
    nyquist_knots = int(np.ceil((2.0 * delta_k * sargs.rbkg.value) / np.pi))
    
    if sargs.nknots is not None:
        nknots = sargs.nknots
        if nknots > nyquist_knots:
            log.warning(
                f'Specified nknots ({nknots}) exceeds the Nyquist limit ({nyquist_knots}) '
                f'for Rbkg={sargs.rbkg.value} and Delta k={delta_k:.2f}. '
                'This may result in overfitting the structural EXAFS oscillations.'
            )
    else:
        nknots = nyquist_knots
        log.debug(f'Calculated {nknots} knots based on Nyquist limit.')

    log.debug('Preparing AUTOBK fitter')

    autobk_fitter = AutobkFitter(
        rbkg=sargs.rbkg.value,
        kweight=sargs.kweight,
        nknots=nknots,
        k_range=k_range
    )

    page_fulldata: dict[str, npt.NDArray[np.floating]] = {
        name: page.domains[Domain.RECIPROCAL].get_columns_data(['k', 'chi']).to_numpy()
        for name, page in context.datastore.pages.items()
    }

    # Data validation checks
    for name, xy in page_fulldata.items():
        x = xy[:,0]
        idx = (x >= k_range[0]) & (x <= k_range[1])
        points_in_range = np.sum(idx)

        if points_in_range <= autobk_fitter.nknots:
            log.warning(
                f'Page {name} has only {points_in_range} data points within the range, '
                f'which is less than or equal to the number of knots ({autobk_fitter.nknots}). '
                'The optimization will likely fail due to insufficient degrees of freedom.'
            )
        elif points_in_range <= autobk_fitter.nknots * 2:
            log.warning(
                f'Page {name} has only {points_in_range} data points within the range, '
                f'which is very close to the number of knots ({autobk_fitter.nknots}). '
                'The optimization may be unstable.'
            )

    compute = partial(_compute_background_autobk, range=k_range, sargs=sargs, autobk_fitter=autobk_fitter, log=log)

    threaded = len(context.datastore.pages) >= 12 and context.options.debug is False
    page_background = execute_threaded(compute, page_fulldata, argkind='s', threaded=threaded, pass_key_as='name')

    log.info('Completed AUTOBK background calculation for all pages.')

    return page_background
