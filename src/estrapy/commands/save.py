import numpy as np
from numpy import typing as npt

from lark import Token, Tree
from pathlib import Path
from dataclasses import dataclass
from typing import Self, Any

from .. import __version__
from ..core.context import Command, CommandResult
from ..core.context import Context, ParseContext
from ..core.commandparser import CommandArgumentParser, CommandArguments, field_arg
from ..core.grammar.mathexpressions import Expression
from ..core.datastore import Domain
from ..core.misc import template_replace, infer_axis_domain
from ..core._validators import type_enum, validate_option_in

@dataclass(slots=True)
class SubCommandArguments_Save_Columns(CommandArguments):
    path: str = field_arg(
        flags=['--path', '-p'],
        type=str,
        required=True
    )

    columns: list[Expression[npt.NDArray[np.floating]]] = field_arg(
        flags=['--columns', '-c'],
        type=Expression.compile,
        nargs='+',
        required=True
    )

    select: Expression[npt.NDArray[np.bool_]] | None = field_arg(
        flags=['--select'],
        type=Expression.compile,
        required=False,
        default=None
    )

    domain: Domain = field_arg(
        flags=['--domain', '-d'],
        type=type_enum(Domain),
        required=False,
        default=Domain.RECIPROCAL,
        help="Domain to use for column expressions. If not specified, the default domain is used.",
        validate=validate_option_in(Domain)
    )

@dataclass(slots=True)
class SubCommandArguments_Table(CommandArguments):
    path: str = field_arg(
        flags=['--path', '-p'],
        type=str,
        required=True
    )

    column: Expression[npt.NDArray[np.floating]] = field_arg(
        flags=['--column', '-c'],
        type=Expression.compile,
        required=True
    )

    axis: Expression[npt.NDArray[np.integer]] = field_arg(
        flags=['--axis', '-a'],
        type=Expression.compile,
        required=True,
    )

    select: Expression[npt.NDArray[np.bool_]] | None = field_arg(
        flags=['--select'],
        type=Expression.compile,
        required=False,
        default=None
    )

    domain: Domain = field_arg(
        flags=['--domain', '-d'],
        type=type_enum(Domain),
        required=False,
        default=None,
        help="Domain to use for column and axis expressions. If not specified, the default domain is used.",
        validate=validate_option_in(Domain)
    )

    def __post_init__(self) -> None:
        _, domain = infer_axis_domain(columns=[*self.axis.required_vars], range=None, domain=self.domain)
        self.domain = domain

@dataclass(slots=True)
class CommandResult_Save(CommandResult):
    ...

@dataclass(slots=True)
class CommandArguments_Save(CommandArguments):
    mode: SubCommandArguments_Save_Columns | SubCommandArguments_Table = field_arg(
        subparsers={
            'columns': SubCommandArguments_Save_Columns,
            'table': SubCommandArguments_Table,
        }
    )

parse_save_command = CommandArgumentParser(CommandArguments_Save, 'save')

def save_to_file(filepath: Path, data: npt.NDArray[Any], columns: list[str], header: list[str]) -> None:

    strdata = np.char.mod('%10.6f', data)
    # The column width is determined by the maximum length of data in each column plus some padding
    colwidth = max(int(np.char.str_len(strdata).max()), max(len(col) for col in columns)) + 1
    headerlines = header + ["  ".join(c.ljust(colwidth) for c in columns)]

    # Ensure path exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    strdata = np.char.ljust(strdata, colwidth)
    
    with open(filepath, 'wt', encoding='utf-8') as f:
        f.writelines('# ' + line + '\n' for line in headerlines)
        f.writelines('  ' + " ".join(row) + '\n' for row in strdata)
    pass


@dataclass(slots=True)
class Command_Save(Command[CommandArguments_Save, CommandResult_Save]):
    @classmethod
    def parse(
        cls: type[Self], commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext
    ) -> Self:
        # TODO: add a way to know at parse time which variables will exist
        arguments = parse_save_command.parse(commandtoken, tokens)
        
        return cls(
            line=commandtoken.line or -1,
            name=commandtoken.value,
            args=arguments,
        )
    def _execute_columns(self, context: Context, args: SubCommandArguments_Save_Columns) -> None:
        log = context.logger.getChild("command.save.columns")
        # Parse column as expressions. The expressions determine expressions per exported column.
        selected_columns = set[str].union(*[expr.required_vars for expr in args.columns])
        if not selected_columns:
            raise ValueError("No columns selected for saving.")
        if args.select is not None:
            selected_columns.update(args.select.required_vars)

        column_names = [expr.to_string() for expr in args.columns]
        
        for _, page in context.datastore.pages.items():
            filename = template_replace(args.path, page.meta)
            filepath = context.paths.outputdir / filename

            domain = page.domains[args.domain]

            data_cols = {str(n):s.to_numpy() for n,s in domain.get_columns_data(list(selected_columns)).items()}
            
            columns: npt.NDArray[Any] = np.column_stack([
                col_expr(**data_cols)
                for col_expr in args.columns
            ])

            if args.select is not None:
                select_idx = np.asarray(args.select(**data_cols), dtype=bool)
            else:
                select_idx = np.ones(columns.shape[0], dtype=bool)


            headerlines: list[str] = []
            headerlines.append(f"Estrapy output file, EstraPy version {__version__}")
            headerlines.append(f"Analyzed with procedure \"{context.projectname}\"")
            headerlines.append(f"Original file: {page.meta['.fn']}")
            headerlines.append(f"Analysis date: {context.starttime.isoformat(sep=' ', timespec='seconds')}")
            headerlines.extend([f"V {k} {v}" for k,v in page.meta._dict.items() if not k.startswith('.')])  # pyright: ignore[reportPrivateUsage]
            headerlines.extend([line.removeprefix("#") for line in page.meta.header.splitlines() if line.startswith("#")])
            
            save_to_file(filepath, columns[select_idx,:], column_names, headerlines)
            log.debug(f"Saved file '{filepath}' with {np.sum(select_idx)} rows.")

        log.info(f"Saved columns to files matching pattern '{args.path}'")

    def _execute_table(self, context: Context, args: SubCommandArguments_Table) -> None:
        log = context.logger.getChild("command.save.table")
        
        selected_columns = set[str].union(args.column.required_vars, args.axis.required_vars)

        if not selected_columns:
            raise ValueError("No columns selected for saving.")
        if args.select is not None:
            selected_columns.update(args.select.required_vars)

        exports: dict[str, tuple[npt.NDArray[Any], dict[str, npt.NDArray[Any]]]] = {}

        for name, page in context.datastore.pages.items():
            filename = template_replace(args.path, page.meta)
            
            domain = page.domains[args.domain]
            data_cols = {str(n):s.to_numpy() for n,s in domain.get_columns_data(list(selected_columns)).items()}

            axis_data = args.axis(**data_cols)
            column_data = args.column(**data_cols)

            if args.select is not None:
                select_idx = np.asarray(args.select(**data_cols), dtype=bool)
                axis_data = axis_data[select_idx]
                column_data = column_data[select_idx]

            if filename not in exports:
                exports[filename] = (axis_data, {name: column_data})
            else:
                if not np.array_equal(exports[filename][0], axis_data):
                    raise ValueError(f"Axis data mismatch for file '{filename}'. Cannot save as table.")
                exports[filename][1][name] = column_data
            
        for filename, (axis, columns) in exports.items():
            filepath = context.paths.outputdir / filename

            axis_name = args.axis.to_string()
            column_name = args.column.to_string()

            headerlines: list[str] = []
            headerlines.append(f"Estrapy output file, EstraPy version {__version__}")
            headerlines.append(f"Analyzed with procedure \"{context.projectname}\"")
            headerlines.append(f"Analysis date: {context.starttime.isoformat(sep=' ', timespec='seconds')}")
            headerlines.append(f"Axis: {axis_name}")
            headerlines.append(f"Column kind: {column_name}")
            headerlines.append( "Original files:")
            headerlines.extend(("  - " + str(page.meta.get('.fn'))) for page in context.datastore.pages.values())

            columns_names = list(columns.keys())
            columns_data = np.column_stack([columns[name] for name in columns_names])

            save_to_file(filepath, np.column_stack([axis, columns_data]), [axis_name] + columns_names, headerlines)
            log.debug(f"Saved file '{filepath}' with {columns_data.shape[1]} columns.")

            

            


    def execute(self, context: Context) -> CommandResult_Save:
        match self.args.mode:
            case SubCommandArguments_Save_Columns() as args:
                self._execute_columns(context, args)
            case SubCommandArguments_Table() as args:
                self._execute_table(context, args)
            case _:
                raise ValueError("Invalid save mode.")
        
        return CommandResult_Save()