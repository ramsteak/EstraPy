import numpy as np

from numpy import typing as npt
from logging import Logger
from scipy.interpolate import make_interp_spline, BSpline # type: ignore missing stub
from dataclasses import dataclass
from lark import Token, Tree
from typing import Self, Callable
from functools import partial

from ..core._validators import validate_number_unit, validate_number_positive, validate_int_non_negative, validate_range_unit

from ..core.context import Command, CommandResult
from ..core.context import Context, ParseContext
from ..core.datastore import Domain, DataPage, ColumnDescription, ColumnKind
from ..core.number import Number, parse_number, parse_range, Unit, parse_edge
from ..core.threaded import execute_threaded
from ..core.commandparser2 import CommandArgumentParser, CommandArguments, field_arg
from ..operations.edge_detection import correlation_edge_detection, SlidingL2Result


@dataclass(slots=True)
class SubCommandArguments_Align_Shift(CommandArguments):
    range: tuple[Number, Number] = field_arg(
        position = 0,
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


@dataclass(slots=True)
class CommandArguments_Align(CommandArguments):
    mode: SubCommandArguments_Align_Calc | SubCommandArguments_Align_Shift = field_arg(
        subparsers = {
            'calc': SubCommandArguments_Align_Calc,
            'shift': SubCommandArguments_Align_Shift,
        }
    )

parse_align_command = CommandArgumentParser(CommandArguments_Align, name='align')

@dataclass(slots=True)
class SubCommandResult_Align_Shift(CommandResult):
    e_axis: npt.NDArray[np.floating]
    average: npt.NDArray[np.floating]
    data: dict[str, npt.NDArray[np.floating]]
    results: dict[str, SlidingL2Result]

    def plot_histogram(self) -> Callable[..., None]:
        """Factory for a callback that plots a histogram of the calculated shifts."""
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes
        
        def plot_histogram_callback(ax: Axes, fig: Figure) -> None:
            shifts = [result.x for result in self.results.values()]
            ax.hist(shifts, bins='sqrt')
            ax.set_title('Histogram of Calculated Shifts')
            ax.set_xlabel('Shift (eV)')
            ax.set_ylabel('Frequency')
        return plot_histogram_callback
    
    def plot_shifts(self) -> Callable[..., None]:
        """Factory for a callback that plots the calculated shifts."""
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        def plot_shifts_callback(ax: Axes, fig: Figure) -> None:
            shifts = [result.x for result in self.results.values()]
            ax.plot(range(len(shifts)), shifts, 'o', alpha=0.7)
            ax.set_title('Calculated Shifts per Spectrum')
            ax.set_xlabel('Spectrum Index')
            ax.set_ylabel('Shift (eV)')
        return plot_shifts_callback
    
    def plot_l2norms(self) -> Callable[..., None]:
        """Factory for a callback that plots the L2 norms for each spectrum."""
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        def plot_l2norms_callback(ax: Axes, fig: Figure) -> None:
            for name, result in self.results.items():
                ax.plot(result.shifts, result.l2_values, label=name)
            ax.set_title('L2 Norms for Each Spectrum')
            ax.set_xlabel('Shift (eV)')
            ax.set_ylabel('L2 Norm')
        return plot_l2norms_callback

    def plot_spectra(self) -> Callable[..., None]:
        """Factory for a callback that plots the interpolated spectra before and after alignment."""
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        def plot_spectra_callback(ax: Axes, fig: Figure) -> None:
            for name, data in self.data.items():
                x = self.results[name].x
                ax.plot(self.e_axis, data, color = 'tab:blue')
                ax.plot(self.e_axis - x, data, color = 'tab:orange')
            
            ax.plot(self.e_axis, self.average, color='black', linewidth=2, linestyle="dotted")
            ax.set_title('Interpolated Spectra')
            ax.set_xlabel('Energy Index')
            ax.set_ylabel('Intensity')
        return plot_spectra_callback

    def plot(self) -> Callable[..., None]:
        """Factory for a callback that plots all the relevant plots in a 2x2 grid."""
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes
        from matplotlib.gridspec import GridSpecFromSubplotSpec

        def plot_callback(ax: Axes, fig: Figure) -> None:
            # Subplot ax with gridspec, and add all the other 4 plots in the quadrants.
            spec = ax.get_subplotspec()
            grid = GridSpecFromSubplotSpec(2, 2, subplot_spec=spec, wspace=0.3, hspace=0.3)
            axes = [
                fig.add_subplot(grid[0,0]),
                fig.add_subplot(grid[0,1]),
                fig.add_subplot(grid[1,0]),
                fig.add_subplot(grid[1,1]),
            ]

            self.plot_histogram()(axes[0], fig)
            self.plot_shifts()(axes[1], fig)
            self.plot_l2norms()(axes[2], fig)
            self.plot_spectra()(axes[3], fig)

            ax.axis('off')
        
        return plot_callback

CommandResult_Align = SubCommandResult_Align_Shift

def _get_shift_axis(range: tuple[float, float], resolution: float) -> npt.NDArray[np.floating]:
    return np.arange(range[0], range[1] + resolution, resolution)

def _get_interpolator_splines_for_pages(pages: dict[str, DataPage], range: tuple[float, float], dom: Domain, xcol:str, ycol:str, log: Logger) -> dict[str, BSpline]:
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

def _apply_shift_to_page(page: DataPage, edge_energy: float | None, shift_energy: float) -> None:
    if edge_energy is not None:
        page.meta['refE0'] = edge_energy

    domain = page.domains[Domain.RECIPROCAL]
    E_column = ColumnDescription('E', Unit.EV, ColumnKind.AXIS, deps=['E'], calc=lambda df, shift=shift_energy: df['E'] - shift, labl='Energy [eV]')
    domain.add_column('E', E_column)

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
                return self._execute_shift(self.args.mode, context)
            case _:
                raise NotImplementedError(f"Unknown mode {self.args.mode} in align command.")
    
    def _execute_shift(self, args: SubCommandArguments_Align_Shift, context: Context) -> SubCommandResult_Align_Shift:
        log = context.logger.getChild('command.align.shift')

        log.debug(f'Aligning spectra with correlation method in range [{args.range[0]!s}, {args.range[1]!s}], resolution {args.resolution!s}, shift {args.shift!s}, derivative {args.derivative}, energy {args.energy}')

        range = args.range[0].value - args.shift.value, args.range[1].value + args.shift.value
        new_e = _get_shift_axis(range, args.resolution.value)

        log.debug('Preparing interpolator splines for all pages.')
        page_splines = _get_interpolator_splines_for_pages(context.datastore.pages, range, Domain.RECIPROCAL, 'E', 'ref', log)

        log.debug('Generating interpolated data for all pages.')
        page_interpolated: dict[str, npt.NDArray[np.floating]] = {name: spline(new_e) for name, spline in page_splines.items()}

        log.debug('Calculating average reference spectrum for alignment.')
        average_reference = np.average([*page_interpolated.values()], axis=0)

        slide_amount = int(args.shift.value // args.resolution.value)
        compute = partial(_compute_l2_shift_from_data,
                        reference = average_reference,
                        derivative = args.derivative,
                        slide_amount = slide_amount,
                        resolution = args.resolution.value,
                        log = log
                    )
        threaded = len(context.datastore.pages) >= 24 and context.options.debug is False
        log.debug(f'Executing shift calculations {"with" if threaded else "without"} threading for {len(context.datastore.pages)} pages.')
        slidenorm = execute_threaded(compute, page_interpolated, argkind = 's', threaded = threaded, pass_key_as='name')

        refE0 = args.energy.value if args.energy is not None else None
        for name,result in slidenorm.items():
            _apply_shift_to_page(context.datastore.pages[name], refE0, result.x)
            log.debug(f'Applied shift of {result.x:0.4f} eV to page {name}.')
        
        log.info('Aligned all spectra using correlation method.')

        return SubCommandResult_Align_Shift(
            e_axis = new_e,
            average = average_reference,
            data = page_interpolated,
            results = slidenorm
        )
