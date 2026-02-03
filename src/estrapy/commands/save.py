import numpy as np
from numpy import typing as npt

from lark import Token, Tree
from pathlib import Path
from dataclasses import dataclass
from typing import Self, Any

from .. import __version__
from ..core.context import Command, CommandResult
from ..core.context import Context, ParseContext
from ..core.commandparser2 import CommandArgumentParser, CommandArguments, field_arg
from ..core.grammar.mathexpressions import Expression
from ..core.datastore import Domain
from ..core.misc import template_replace

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

@dataclass(slots=True)
class CommandResult_Save(CommandResult):
    ...

@dataclass(slots=True)
class CommandArguments_Save(CommandArguments):
    mode: SubCommandArguments_Save_Columns = field_arg(
        subparsers={
            'columns': SubCommandArguments_Save_Columns,
        }
    )

parse_save_command = CommandArgumentParser(CommandArguments_Save, 'save')

def save_to_file(filepath: Path, data: npt.NDArray[Any], columns: list[str], header: str) -> None:

    strdata = np.char.mod('%10.6f', data)
    # The column width is determined by the maximum length of data in each column plus some padding
    colwidth = max(int(np.char.str_len(strdata).max()), max(len(col) for col in columns)) + 1
    header += "# " + "  ".join(c.ljust(colwidth) for c in columns) + "\n"

    # Ensure path exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    strdata = np.char.ljust(strdata, colwidth)
    
    with open(filepath, 'wt', encoding='utf-8') as f:
        f.write(header)
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

            domain = page.domains[Domain.RECIPROCAL] # TODO: make domain selectable            

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
            headerlines.append(f"# Estrapy output file, EstraPy version {__version__}")
            headerlines.append(f"# Analyzed with procedure \"{context.projectname}\"")
            headerlines.append(f"# Original file: {page.meta['.fn']}")
            headerlines.append(f"# Analysis date: {context.starttime.isoformat(sep=' ', timespec='seconds')}")
            headerlines.extend([f"#V {k} {v}" for k,v in page.meta._dict.items() if not k.startswith('.')])  # pyright: ignore[reportPrivateUsage]
            header = "\n".join(headerlines)
            header += f"\n{page.meta.header}"
            header += '\n' if not header.endswith('\n') else ''

            save_to_file(filepath, columns[select_idx,:], column_names, header)
            log.debug(f"Saved file '{filepath}' with {np.sum(select_idx)} rows.")

        log.info(f"Saved columns to files matching pattern '{args.path}'")


    def execute(self, context: Context) -> CommandResult_Save:
        match self.args.mode:
            case SubCommandArguments_Save_Columns() as args:
                self._execute_columns(context, args)
        
        return CommandResult_Save()