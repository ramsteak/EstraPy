import numpy as np

from numpy import typing as npt
from logging import Logger
from scipy.interpolate import make_interp_spline, BSpline
from dataclasses import dataclass
from typing import Callable
from functools import partial

from .result import CommandResult_Align, apply_shift_to_page

from ...core._validators import validate_number_unit, validate_number_positive, validate_int_non_negative, validate_range_unit
from ...core.context import Context
from ...core.datastore import Domain, DataPage
from ...core.number import Number, parse_number, parse_range, Unit, parse_edge
from ...core.threaded import execute_threaded
from ...core.commandparser import CommandArguments, field_arg
from ...operations.edge_detection import correlation_edge_detection, SlidingL2Result


@dataclass(slots=True)
class SubCommandArguments_Align_Shift(CommandArguments):
    range: tuple[Number, Number] = field_arg(
        types = parse_range,
        nargs = 2,
        required = False,
        help = 'Energy range to consider for alignment.',
        default = (Number(None, -np.inf, None), Number(None, np.inf, None)),
        validate = validate_range_unit(Unit.EV)
    )

    resolution: Number = field_arg(
        flags = ['--resolution', '--res'],
        type = parse_number,
        required = True,
        help = 'Energy resolution for the alignment calculation.',
        validate = [validate_number_unit(Unit.EV), validate_number_positive],
        default = Number(None, 0.1, Unit.EV)
    )

    shift: Number = field_arg(
        flags = ['--shift', '-s'],
        type = parse_number,
        required = True,
        help = 'Maximum shift to consider for the alignment calculation.',
        default = Number(None, 5.0, Unit.EV),
        validate = [validate_number_unit(Unit.EV), validate_number_positive],
    )

    derivative: int = field_arg(
        flags = ['--derivative', '--deriv'],
        type = int,
        required = False,
        help = 'Derivative order to use for the correlation calculation.',
        default = 0,
        validate = validate_int_non_negative,
    )

    energy: Number | None = field_arg(
        flags = ['--energy', '--E0', '-E'],
        type = parse_edge,
        required = False,
        help = 'Edge energy to set in the metadata after alignment.',
        default = None,
        validate = validate_number_unit(Unit.EV),
    )


@dataclass(slots=True)
class SubCommandResult_Align_Shift(CommandResult_Align):
    e_axis: npt.NDArray[np.floating]          # New energy axis for the aligned spectra
    average: npt.NDArray[np.floating]         # Average reference spectrum used for alignment
    data: dict[str, npt.NDArray[np.floating]] # Interpolated spectra for each page, keyed by page name
    results: dict[str, SlidingL2Result]       # Result of the L2 shift calculation for each page

    def plot_histogram(self) -> Callable[..., None]:
        """Factory for a callback that plots a histogram of the calculated shifts."""
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes
        
        def plot_histogram_callback(ax: Axes, fig: Figure) -> None:
            shifts = [result.x for result in self.results.values()]
            ax.hist(shifts, bins='sqrt') # pyright: ignore[reportUnknownMemberType]
            ax.set_title('Histogram of Calculated Shifts') # pyright: ignore[reportUnknownMemberType]
            ax.set_xlabel('Shift (eV)') # pyright: ignore[reportUnknownMemberType]
            ax.set_ylabel('Frequency') # pyright: ignore[reportUnknownMemberType]
        return plot_histogram_callback
    
    def plot_shifts(self) -> Callable[..., None]:
        """Factory for a callback that plots the calculated shifts."""
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        def plot_shifts_callback(ax: Axes, fig: Figure) -> None:
            shifts = [result.x for result in self.results.values()]
            ax.plot(range(len(shifts)), shifts, 'o', alpha=0.7) # pyright: ignore[reportUnknownMemberType]
            ax.set_title('Calculated Shifts per Spectrum') # pyright: ignore[reportUnknownMemberType]
            ax.set_xlabel('Spectrum Index') # pyright: ignore[reportUnknownMemberType]
            ax.set_ylabel('Shift (eV)') # pyright: ignore[reportUnknownMemberType]
        return plot_shifts_callback
    
    def plot_l2norms(self) -> Callable[..., None]:
        """Factory for a callback that plots the L2 norms for each spectrum."""
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        def plot_l2norms_callback(ax: Axes, fig: Figure) -> None:
            for name, result in self.results.items():
                ax.plot(result.shifts, result.l2_values, label=name) # pyright: ignore[reportUnknownMemberType]
            ax.set_title('L2 Norms for Each Spectrum') # pyright: ignore[reportUnknownMemberType]
            ax.set_xlabel('Shift (eV)') # pyright: ignore[reportUnknownMemberType]
            ax.set_ylabel('L2 Norm') # pyright: ignore[reportUnknownMemberType]
        return plot_l2norms_callback

    def plot_spectra(self) -> Callable[..., None]:
        """Factory for a callback that plots the interpolated spectra before and after alignment."""
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        def plot_spectra_callback(ax: Axes, fig: Figure) -> None:
            for name, data in self.data.items():
                x = self.results[name].x
                ax.plot(self.e_axis, data, color = 'tab:blue') # pyright: ignore[reportUnknownMemberType]
                ax.plot(self.e_axis - x, data, color = 'tab:orange') # pyright: ignore[reportUnknownMemberType]
            
            ax.plot(self.e_axis, self.average, color='black', linewidth=2, linestyle="dotted") # pyright: ignore[reportUnknownMemberType]
            ax.set_title('Interpolated Spectra') # pyright: ignore[reportUnknownMemberType]
            ax.set_xlabel('Energy Index') # pyright: ignore[reportUnknownMemberType]
            ax.set_ylabel('Intensity') # pyright: ignore[reportUnknownMemberType]
        return plot_spectra_callback

    def plot(self) -> Callable[..., None]:
        """Factory for a callback that plots all the relevant plots in a 2x2 grid."""
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes
        from matplotlib.gridspec import GridSpecFromSubplotSpec

        def plot_callback(ax: Axes, fig: Figure) -> None:
            # Subplot ax with gridspec, and add all the other 4 plots in the quadrants.
            spec = ax.get_subplotspec() # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            grid = GridSpecFromSubplotSpec(2, 2, subplot_spec=spec, wspace=0.3, hspace=0.3)
            axes = [
                fig.add_subplot(grid[0,0]), # pyright: ignore[reportUnknownMemberType]
                fig.add_subplot(grid[0,1]), # pyright: ignore[reportUnknownMemberType]
                fig.add_subplot(grid[1,0]), # pyright: ignore[reportUnknownMemberType]
                fig.add_subplot(grid[1,1]), # pyright: ignore[reportUnknownMemberType]
            ]

            self.plot_histogram()(axes[0], fig)
            self.plot_shifts()(axes[1], fig)
            self.plot_l2norms()(axes[2], fig)
            self.plot_spectra()(axes[3], fig)

            ax.axis('off') # pyright: ignore[reportUnknownMemberType]
        
        return plot_callback

def _generate_shift_axis(range: tuple[float, float], resolution: float) -> npt.NDArray[np.floating]:
    """Generates the new energy axis for the shifted spectra based on the specified range and resolution."""
    return np.arange(range[0], range[1] + resolution, resolution)


def _make_interpolator_splines_for_pages(pages: dict[str, DataPage], range: tuple[float, float], dom: Domain, xcol:str, ycol:str, log: Logger) -> dict[str, BSpline]:
    """Prepares the interpolator splines for all pages in the specified domain and columns, within the given energy range."""
    splines: dict[str, BSpline] = {}
    for name, page in pages.items():
        domain = page.domains[dom]
        df = domain.get_columns_data([xcol, ycol])
        index = (df[xcol] >= range[0]) & (df[xcol] <= range[1])

        region = df[index]
        if not region.size:
            log.warning(f'No data in the specified range for page {name}. It will be skipped, this may lead to further errors. Consider adjusting the range.')
            continue

        spline:BSpline = make_interp_spline(region[xcol], region[ycol], k=3) # type: ignore
        spline.extrapolate = True

        splines[name] = spline

    return splines

def _compute_l2_shift_from_data(data: npt.NDArray[np.floating], name: str, reference: npt.NDArray[np.floating], derivative: int, slide_amount: int, resolution: float, log: Logger) -> SlidingL2Result:
    result = correlation_edge_detection(data, reference, derivative, slide_amount, resolution)
    log.debug(f'Calculated shift for page {name}: {result.x:0.4f} eV')
    return result


def execute_shift(context: Context, sargs: SubCommandArguments_Align_Shift) -> SubCommandResult_Align_Shift:
    log = context.logger.getChild('command.align.shift')

    log.debug(f'Aligning spectra with correlation method in range [{sargs.range[0]!s}, {sargs.range[1]!s}], resolution {sargs.resolution!s}, shift {sargs.shift!s}, derivative {sargs.derivative}, energy {sargs.energy}')

    range = sargs.range[0].value - sargs.shift.value, sargs.range[1].value + sargs.shift.value
    new_e = _generate_shift_axis(range, sargs.resolution.value)

    log.debug('Preparing interpolator splines for all pages.')
    page_splines = _make_interpolator_splines_for_pages(context.datastore.pages, range, Domain.RECIPROCAL, 'E', 'ref', log)

    log.debug('Generating interpolated data for all pages.')
    page_interpolated: dict[str, npt.NDArray[np.floating]] = {name: spline(new_e) for name, spline in page_splines.items()}

    log.debug('Calculating average reference spectrum for alignment.')
    average_reference = np.average([*page_interpolated.values()], axis=0)

    slide_amount = int(sargs.shift.value // sargs.resolution.value)
    compute = partial(_compute_l2_shift_from_data,
                    reference = average_reference,
                    derivative = sargs.derivative,
                    slide_amount = slide_amount,
                    resolution = sargs.resolution.value,
                    log = log
                )
    threaded = len(context.datastore.pages) >= 24 and context.options.debug is False
    log.debug(f'Executing shift calculations {"with" if threaded else "without"} threading for {len(context.datastore.pages)} pages.')
    slidenorm = execute_threaded(compute, page_interpolated, argkind = 's', threaded = threaded, pass_key_as='name')

    refE0 = sargs.energy.value if sargs.energy is not None else None
    for name,result in slidenorm.items():
        apply_shift_to_page(context.datastore.pages[name], refE0, result.x)
        log.debug(f'Applied shift of {result.x:0.4f} eV to page {name}.')
    
    log.info('Aligned all spectra using correlation method.')

    return SubCommandResult_Align_Shift(
        e_axis = new_e,
        average = average_reference,
        data = page_interpolated,
        results = slidenorm
    )
