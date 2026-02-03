import numpy as np

from lark import Token, Tree

from dataclasses import dataclass
from typing import Self

from ..core.context import Command, CommandResult
from ..core.context import Context, ParseContext
from ..core.commandparser import CommandArgumentParser, CommandArguments, field_arg
from ..core._validators import validate_number_positive, validate_number_unit, validate_range_unit, validate_float_non_negative, type_fuzzy, validate_option_in
from ..core.number import Number, parse_range, parse_number, Unit
from ..operations.fourier import fourier, apodizer_functions, flattop_window
from ..operations.axis_conversions import Nyquist
from ..core.datastore import Domain, ColumnKind, ColumnDescription, DataDomain
from ..core.misc import fmt

@dataclass(slots=True)
class CommandArguments_Fourier(CommandArguments):
    range: tuple[Number, Number] = field_arg(
        position=0,
        types=parse_range,
        nargs=2,
        required=True,
        validate=validate_range_unit(Unit.K)
    )

    maxR: Number | None = field_arg(
        position=1,
        type=parse_number,
        required=False,
        default=None,
        validate=[validate_number_unit(Unit.A), validate_number_positive]
    )

    spacing: Number | None = field_arg(
        flags=['--dr', '--spacing'],
        type=parse_number,
        required=False,
        default=None,
        validate=[validate_number_unit(Unit.A), validate_number_positive]
    )

    kweight: float = field_arg(
        flags=['--kweight', '-k'],
        type=float,
        required=False,
        default=0.0,
        validate=validate_float_non_negative
    )

    width: Number | None = field_arg(
        flags=['--width', '-w'],
        type=parse_number,
        required=False,
        default=None,
        validate=[validate_number_unit(Unit.A), validate_number_positive]
    )

    apodizer: str = field_arg(
        flags=['--apodizer', '-a'],
        type=type_fuzzy(list(apodizer_functions)),
        required=False,
        default='hanning',
        validate=validate_option_in(apodizer_functions)
    )

    apodizerp: float = field_arg(
        flags=['--parameter', '-p'],
        type=float,
        required=False,
        default=3.0
    )

parse_fourier_command = CommandArgumentParser(CommandArguments_Fourier, 'fourier')

class CommandResult_Fourier(CommandResult):
    ...

@dataclass(slots=True)
class Command_Fourier(Command[CommandArguments_Fourier, CommandResult_Fourier]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        arguments = parse_fourier_command.parse(commandtoken, tokens)

        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )

    def execute(self, context: Context) -> CommandResult_Fourier:
        log = context.logger.getChild('command.fourier')

        for name, page in context.datastore.pages.items():
            domain = page.domains[Domain.RECIPROCAL]

            if Domain.FOURIER not in page.domains:
                newdom = DataDomain()
                page.domains[Domain.FOURIER] = newdom
            
            fourier_domain = page.domains[Domain.FOURIER]
            
            # Get the valid range
            range = self.args.range[0].value, self.args.range[1].value
            df = domain.get_columns_data(['k', 'chi'])  # Ensure columns exist
            idx = (df['k'] >= range[0]) & (df['k'] <= range[1])

            # Define the new r axis
            k_axis = domain.get_column_data('k')[idx].to_numpy()
            match self.args.spacing, self.args.maxR:
                case None, None:
                    r_axis = Nyquist.axis(k_axis)
                case None, Number(value=maxR):
                    _, diff = Nyquist.span_diff(k_axis)
                    r_axis = np.arange(0.0, maxR + diff, diff)
                case Number(value=spacing), None:
                    span,_ = Nyquist.span_diff(k_axis)
                    r_axis = np.arange(0.0, span + spacing, spacing)
                case Number(value=spacing), Number(value=maxR):
                    r_axis = np.arange(0.0, maxR + spacing, spacing)

            ramp_width = self.args.width.value if self.args.width is not None else 0.5 * (range[1] - range[0])
            w = flattop_window(k_axis, (range[0], range[0]+ramp_width, range[1]-ramp_width, range[1]), self.args.apodizer)
            chi = df['chi'][idx].to_numpy() * w * k_axis**self.args.kweight
            f = fourier(k_axis, chi, r_axis)

            fourier_domain.add_column_data(
                'r',
                ColumnDescription(
                    name='r',
                    type=ColumnKind.AXIS,
                    unit=Unit.A,
                    labl='Radial distance [Å]'
                ),
                r_axis
            )
            fourier_domain.add_column_data(
                'f',
                ColumnDescription(
                    name='f',
                    type=ColumnKind.DATA,
                    unit=None,
                    labl=f'FT{{k^{self.args.kweight}*χ(k)}} [Å⁻{fmt.sup(int(self.args.kweight-1))}]'
                ),
                f
            )
            log.debug(f'Computed Fourier transform for page "{name}" with {len(r_axis)} points.')
        log.info(f'Computed Fourier transform for {len(context.datastore.pages)} pages.')
        
        return CommandResult_Fourier()