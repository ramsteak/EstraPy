from typing import NamedTuple
from logging import getLogger

from . import __version__
from .commands import commands
from .commands._context import Directives, Context
from .commands._handler import CommandHandler
from .commands._parser import InputFileParsingException

VERSION = tuple(int(i) for i in __version__.split("."))


def parse_version(input: str) -> tuple[int, ...]:
    # At most the version line will be 25 characters long. This is in order to
    # reduce the amount of unnecessary string splitting.
    firstline = input[:25].splitlines()[0]
    if not firstline.startswith("#"):
        raise InputFileParsingException(
            "The first line of the input file must declare the version."
        )

    versionstr = (
        firstline.removeprefix("#")
        .strip()
        .removeprefix("version")
        .removeprefix(":")
        .strip()
    )

    return tuple(int(d) for d in versionstr.split("."))


def parse_directives(input: str) -> Directives:
    lines = input.splitlines()
    # Directives are prefixed by %, and follow the version. Only empty lines can
    # be between directives or version, and all directives must be at the top of
    # the file.

    clear = False
    noplot = False
    vars = {}

    for line in lines[1:]:
        # Skip empty lines
        if not line.strip():
            continue
        # Stop directive parsing after the first command line
        if not line.startswith("%"):
            break
        
        directive = line.removeprefix("%").strip()
        if directive == "clear":
            clear = True
        if directive == "noplot":
            noplot = True

        if directive.startswith("define "):
            vname, vval = directive.removeprefix("define ").split(" ", maxsplit=1)
            vars[vname] = vval

    return Directives(clear, noplot, vars)


def parse_commands(
    input: str, context: Context
) -> list[tuple[int, CommandHandler, NamedTuple]]:
    log = getLogger("parser")

    # Check file version
    if context.options.version[:3] != VERSION[:3]:
        # TODO: is not valueerror
        raise ValueError(
            f"Estrapy version ({VERSION}) does not match file version {context.options.version}"
        )
    
    replaced = input
    for varname, varval in context.directives.vars.items():
        replaced = replaced.replace(f"%{varname}%", varval)
        replaced = replaced.replace(f"${{{varname}}}", varval)

    lines = [*enumerate(replaced.splitlines(), 1)]

    parsedcommands: list[tuple[int, CommandHandler, NamedTuple]] = []

    lineid = 0
    while lineid + 1 < len(lines):
        lineid += 1
        index, line = lines[lineid]
        if not line.strip():
            continue
        if line.startswith("#"):
            continue
        if line.startswith("%"):
            continue

        if line.startswith(" "):
            raise InputFileParsingException("Command cannot start with spaces.")

        cmd = line.split(maxsplit=1)[0]

        if cmd == "exit":
            return parsedcommands
        
        if cmd not in commands:
            raise InputFileParsingException(f"Unrecognized command: {cmd}")

        command = commands[cmd]

        # Check if successive lines start with spaces and are non empty, and add
        # them to the current command line.
        step = 1
        while (
            (lineid + step < len(lines))
            and (lines[lineid + step][1].startswith(" "))
            and lines[lineid + step][1].strip()
        ):
            step += 1

        tokens = command.tokenize([(i, lines[i][1]) for i in range(lineid, lineid + step)])
        # Removes the first token, corresponding to the command name.
        try:
            commandargs = command.parse(tokens[1:], context)
        except Exception as E:
            log.critical(f"Syntax error: line {index} {str(E)}")
            exit(-1)

        parsedcommands.append((index, command, commandargs))

        # -1 is due to the +1 at the beginning of the loop
        lineid += step - 1

    return parsedcommands
