import numpy as np

from numpy import typing as npt
from lark import Token, Tree
from scipy.stats import norm # pyright: ignore[reportMissingTypeStubs]

from dataclasses import dataclass
from typing import Self, Mapping, override

from ..core.datastore import Domain, ColumnKind
from ..core.context import Context, ParseContext, Command, CommandResult
from ..core.commandparser import CommandArgumentParser, CommandArguments, field_arg
from ..core._validators import validate_range_unit, validate_number_unit, validate_int_non_negative, validate_float_positive, validate_float_non_negative
from ..core.number import Number, parse_range, Unit, parse_number
from ..core.misc import infer_axis_domain
from ..operations.robust import robust_polyfit, median_window
from ..operations.evenodd import diff_even


# region Finders

@dataclass(slots=True)
class SubCommandArguments_Finder(CommandArguments):
    def _get_numbers(self) -> list[Number | None] | None:
        """Internal helper to get all Number arguments for axis inference."""
        return None

@dataclass(slots=True)
class SubCommandResult_Finder(CommandResult):
    detect_range: npt.NDArray[np.bool_]
    glitch_region: npt.NDArray[np.bool_]

@dataclass(slots=True)
class SubCommandArguments_Finder_Force(SubCommandArguments_Finder):
    ...

@dataclass
class SubCommandResult_Finder_Force(SubCommandResult_Finder):
    ...

@dataclass(slots=True)
class SubCommandArguments_Finder_Point(SubCommandArguments_Finder):
    position: Number = field_arg(
        position=0,
        type=parse_number,
        required=True,
        validate=validate_number_unit(Unit.K, Unit.EV),
    )
    @override
    def _get_numbers(self) -> list[Number | None] | None:
        return [self.position]

@dataclass(slots=True)
class SubCommandResult_Finder_Point(SubCommandResult_Finder):
    ...


@dataclass(slots=True)
class SubCommandArguments_Finder_Polynomial(SubCommandArguments_Finder):
    degree: int = field_arg(
        flags=['--degree', '-d'],
        type=int,
        required=True,
        const_flags={
            '--linear': 1, '-l': 1,
            '--quadratic': 2, '-q': 2,
            '--cubic': 3, '-c': 3,
            '--constant': 0, '-C': 0,
        },
        validate=validate_int_non_negative,
    )

    pvalue: float = field_arg(
        flags=['--pvalue', '-p'],
        type=float,
        required=False,
        default=0.0002,
        validate=validate_float_positive
    )

    axis: str = field_arg(
        flags=['--axis'],
        type=str,
        required=False,
        default=None,
    )

    column: str = field_arg(
        flags=['--column', '--col'],
        type=str,
        required=False,
        default='I0'
    )


@dataclass(slots=True)
class SubCommandResult_Finder_Polynomial(SubCommandResult_Finder):
    baseline: npt.NDArray[np.floating]
    coeffs: npt.NDArray[np.floating]
    threshold: float


@dataclass(slots=True)
class SubCommandArguments_Finder_Even(SubCommandArguments_Finder):
    column: str = field_arg(
        flags=['--column', '--col'],
        type=str,
        required=False,
        default='I0'
    )

    pvalue: float = field_arg(
        flags=['--pvalue', '-p'],
        type=float,
        required=False,
        default=0.0002,
        validate=validate_float_positive
    )

    window: int = field_arg(
        flags=['--median-window'],
        type=int,
        required=False,
        default=1,
        validate=validate_int_non_negative
    )

@dataclass(slots=True)
class SubCommandResult_Finder_Even(SubCommandResult_Finder):
    diff: npt.NDArray[np.floating]
    threshold: float

# endregion

# region Methods

@dataclass(slots=True)
class SubCommandArguments_Method(CommandArguments):
    ...
@dataclass(slots=True)
class SubCommandResult_Method(CommandResult):
    ...

@dataclass(slots=True)
class SubCommandArguments_Method_Remove(SubCommandArguments_Method):
    ...
@dataclass(slots=True)
class SubCommandResult_Method_Remove(SubCommandResult_Method):
    ...

@dataclass(slots=True)
class SubCommandArguments_Method_Interpolate(SubCommandArguments_Method):
    ...
@dataclass(slots=True)
class SubCommandResult_Method_Interpolate(SubCommandResult_Method):
    ...

# endregion


@dataclass(slots=True)
class CommandArguments_Deglitch(CommandArguments):
    range: tuple[Number, Number] = field_arg(
        position=0,
        types=parse_range,
        nargs=2,
        required=False,
        validate=validate_range_unit(Unit.EV, Unit.K),
        default=(Number(None, -np.inf, None), Number(None, np.inf, None)),
    )

    kweight: float = field_arg(
        flags=['--kweight', '-k'],
        type=float,
        required=False,
        default=0.0,
        validate=validate_float_non_negative
    )

    finder: SubCommandArguments_Finder = field_arg(
        subparsers={
            'force': SubCommandArguments_Finder_Force,
            'polynomial': SubCommandArguments_Finder_Polynomial,
            'point': SubCommandArguments_Finder_Point,
            'even': SubCommandArguments_Finder_Even,
        },
        required=True,
    )

    method: SubCommandArguments_Method = field_arg(
        subparsers={
            'remove': SubCommandArguments_Method_Remove,
            'interpolate': SubCommandArguments_Method_Interpolate,
        },
        required=True,
    )

    def __post_init__(self) -> None:
        # Infer axis from range unit if the subcommand requires it
        # Define a protocol for subcommands that have an axis argument
        if hasattr(self.finder, 'axis'):
            axis, _ = infer_axis_domain(
                domain = Domain.RECIPROCAL,
                axis = getattr(self.finder, 'axis', None),
                range = self.range,
                numbers = self.finder._get_numbers() # pyright: ignore[reportPrivateUsage]
            )
            setattr(self.finder, 'axis', axis)


parse_deglitch_command = CommandArgumentParser(CommandArguments_Deglitch, 'deglitch')



@dataclass(slots=True)
class CommandResult_Deglitch(CommandResult):
    ...


@dataclass(slots=True)
class Command_Deglitch(Command[CommandArguments_Deglitch, CommandResult_Deglitch]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_deglitch_command.parse(commandtoken, tokens)
        
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Deglitch:
        args = self.args

        log = context.logger.getChild('deglitch.finder')


        match args.finder:
            case SubCommandArguments_Finder_Force() as sargs:
                glitches = self._execute_finder_force(context, args.range, sargs)
            case SubCommandArguments_Finder_Point() as sargs:
                glitches = self._execute_finder_point(context, args.range, sargs)
            case SubCommandArguments_Finder_Polynomial() as sargs:
                glitches = self._execute_finder_polynomial(context, args.range, sargs)
            case SubCommandArguments_Finder_Even() as sargs:
                glitches = self._execute_finder_even(context, args.range, sargs)
            case _:
                raise NotImplementedError(f'Finder type "{type(args.finder)}" not implemented.')
        
        # Check the validity of the glitches found
        for page_name, result in glitches.items():
            if np.sum(result.glitch_region) == result.glitch_region.shape[0]:
                raise ValueError(f'Forced glitch in page "{page_name}" covers the entire axis range. This would delete all data points.')
            if np.sum(result.glitch_region) >= result.glitch_region.shape[0] * 0.05:
                log.warning(f'Finder "{type(args.finder).__name__}" found a large glitch region in page "{page_name}" covering {np.sum(result.glitch_region)} points ({(np.sum(result.glitch_region) / result.glitch_region.shape[0]) * 100:.2f} % of total points).')

        match args.method:
            case SubCommandArguments_Method_Remove() as sargs:
                self._execute_method_remove(context, args.range, glitches, sargs)
            case SubCommandArguments_Method_Interpolate() as sargs:
                self._execute_method_interpolate(context, args.range, glitches, sargs)
            case _:
                raise NotImplementedError(f'Method type "{type(args.method)}" not implemented.')
    
        return CommandResult_Deglitch()


    def _execute_finder_force(self, context: Context, range: tuple[Number, Number], args: SubCommandArguments_Finder_Force) -> dict[str, SubCommandResult_Finder_Force]:
        glitches: dict[str, SubCommandResult_Finder_Force] = {}
        axis, _ = infer_axis_domain(domain = Domain.RECIPROCAL, range = range)

        for name, page in context.datastore.pages.items():
            ax = page.domains[Domain.RECIPROCAL].get_column_data(axis).to_numpy()

            forced_glitch = (range[0].value <= ax) & (ax <= range[1].value)

            glitches[name] = SubCommandResult_Finder_Force(
                detect_range=np.ones_like(ax, dtype=bool),
                glitch_region=forced_glitch
            )
        
        return glitches
    
    def _execute_finder_point(self, context: Context, range: tuple[Number, Number], args: SubCommandArguments_Finder_Point) -> dict[str, SubCommandResult_Finder_Point]:
        glitches: dict[str, SubCommandResult_Finder_Point] = {}
        axis, _ = infer_axis_domain(domain = Domain.RECIPROCAL, range = range, numbers = [args.position])
        for name, page in context.datastore.pages.items():
            ax = page.domains[Domain.RECIPROCAL].get_column_data(axis).to_numpy()
            point_glitch = np.zeros_like(ax, dtype=bool)
            point_glitch[np.abs(ax - args.position.value).argmin()] = True
            glitches[name] = SubCommandResult_Finder_Point(
                detect_range = np.ones_like(ax, dtype=bool),
                glitch_region=point_glitch
            )
        
        return glitches
        
    def _execute_finder_polynomial(self, context: Context, range: tuple[Number, Number], args: SubCommandArguments_Finder_Polynomial) -> dict[str, SubCommandResult_Finder_Polynomial]:
        glitches: dict[str, SubCommandResult_Finder_Polynomial] = {}
        axis, _ = infer_axis_domain(domain = Domain.RECIPROCAL, range = range, axis = args.axis)

        clip = float(norm.ppf(1 - args.pvalue/2)) # pyright: ignore[reportUnknownMemberType]

        for name, page in context.datastore.pages.items():
            X = page.domains[Domain.RECIPROCAL].get_column_data(axis).to_numpy()
            Y = page.domains[Domain.RECIPROCAL].get_column_data(args.column).to_numpy()

            # Fit polynomial to the data in the specified range
            idx = (range[0].value <= X) & (X <= range[1].value)
            x, y = X[idx], Y[idx]

            coeffs, baseline, inliers, sigma = robust_polyfit(x, y, deg=args.degree, n_iter=5, clip=clip)
            glitches[name] = SubCommandResult_Finder_Polynomial(
                detect_range = idx,
                glitch_region = ~inliers,
                baseline = baseline,
                coeffs = coeffs,
                threshold = clip * sigma
            )

        return glitches
    
    def _execute_finder_even(self, context: Context, range: tuple[Number, Number], args: SubCommandArguments_Finder_Even) -> dict[str, SubCommandResult_Finder_Even]:
        glitches: dict[str, SubCommandResult_Finder_Even] = {}
        axis, _ = infer_axis_domain(domain = Domain.RECIPROCAL, range = range)
        
        clip = float(norm.ppf(1 - args.pvalue/2)) # pyright: ignore[reportUnknownMemberType]

        for name, page in context.datastore.pages.items():
            X = page.domains[Domain.RECIPROCAL].get_column_data(axis).to_numpy()
            Y = page.domains[Domain.RECIPROCAL].get_column_data(args.column).to_numpy()

            idx = (range[0].value <= X) & (X <= range[1].value)
            x, y = X[idx], Y[idx]

            diff = median_window(diff_even(x, y), window=args.window*2+1)
            # Median-smoothing the diff-even helps reducing the multiple spiking effect of glitches
            # in the even-odd difference noise estimation.

            sigma = 1.4826 * float(np.median(np.abs(diff - np.median(diff)))) * 2.5
            # 2.5 acts as a correction factor for the MAD estimator, that is very sensitive
            # both to the reduced number of points in the even-odd difference and to the presence
            # of glitches in the data.

            glitches[name] = SubCommandResult_Finder_Even(
                detect_range = idx,
                glitch_region = np.abs(diff) > clip * sigma,
                diff = diff,
                threshold = clip * sigma
            )

        return glitches


    def _execute_method_remove(self, context: Context, range: tuple[Number, Number], glitches: Mapping[str, SubCommandResult_Finder], args: SubCommandArguments_Method_Remove) -> SubCommandResult_Method_Remove:
        log = context.logger.getChild('deglitch.method.remove')

        for page_name, result in glitches.items():
            page = context.datastore.pages[page_name]
            domain = page.domains[Domain.RECIPROCAL]

            # Removes data points in glitch region from the dataframe. This operation is irreversible.
            full_index = np.ones(domain.data.shape[0], dtype=bool)
            full_index[result.detect_range] = result.glitch_region
            domain.data = domain.data.loc[~full_index].reset_index(drop=True)
            log.debug(f'Removed {np.sum(result.glitch_region)} data points from page "{page_name}".')
        
        return SubCommandResult_Method_Remove()

    def _execute_method_interpolate(self, context: Context, range: tuple[Number, Number], glitches: Mapping[str, SubCommandResult_Finder], args: SubCommandArguments_Method_Interpolate) -> SubCommandResult_Method_Interpolate:
        log = context.logger.getChild('deglitch.method.interpolate')
        axis, _ = infer_axis_domain(domain = Domain.RECIPROCAL, range = range)

        for page_name, result in glitches.items():
            page = context.datastore.pages[page_name]
            domain = page.domains[Domain.RECIPROCAL]

            full_glitch = np.zeros(domain.data.shape[0], dtype=bool)
            full_glitch[result.detect_range] = result.glitch_region

            for col_name, column_hist in domain.columns.items():
                if column_hist[-1].desc.type != ColumnKind.DATA:
                    continue
                
                X = domain.get_column_data(axis).to_numpy()
                Y = domain.get_column_data(col_name).to_numpy()

                # Interpolate over the glitch region
                Y_interp = np.copy(Y)
                Y_interp[full_glitch] = np.interp(
                    X[full_glitch],
                    X[~full_glitch],
                    Y[~full_glitch]
                )

                # Update the column data
                domain.add_column_data(col_name, column_hist[-1].desc, Y_interp)

            log.debug(f'Interpolated over {np.sum(result.glitch_region)} data points in page "{page_name}".')
        
        return SubCommandResult_Method_Interpolate()
    