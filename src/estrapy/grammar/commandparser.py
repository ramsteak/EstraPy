from typing import TypeVar, Generic, Iterable, Callable, Any
from lark import Token, Tree
from dataclasses import dataclass, fields, MISSING, field

from ..core.grammarclasses import CommandArguments, CommandMetadata, Command
from ..core.errors import ParseError
from ..core.errors import ArgumentError, DuplicateArgumentError
from ..core.context import ParseContext

# Define a command parser that can parse commands and their arguments, akin to an argparser.
_T = TypeVar('_T', bound=CommandArguments)


@dataclass(slots=True)
class ActionSpecification:
    destination: str
    action: str
    constant: Any = None


@dataclass(slots=True)
class ArgumentSpecification:
    name: str
    action: ActionSpecification
    type: Callable[[str], Any] = str
    required: bool = False
    default: Any = None
    nargs: int | str = 1


@dataclass(slots=True)
class OptionSpecification(ArgumentSpecification):
    arg_names: list[str] = field(default_factory=list[str])


@dataclass(slots=True)
class ValueSpecification(ArgumentSpecification):
    position: int = -1


_MISSING_ARG = object()


class CommandParser(Generic[_T]):
    def __init__(self, returnstruct: type[_T], metadata: CommandMetadata, *args: Any, **kwargs: Any) -> None:
        # Check that returnstruct is a dataclass
        if not hasattr(returnstruct, '__dataclass_fields__'):
            raise TypeError('returnstruct must be a dataclass')
        self.returnstruct = returnstruct
        self.structinit = args, kwargs
        self.metadata = metadata
        # Map from option name in the struct to its specification
        # (e.g. 'energy' -> ArgumentSpecification)
        self.arguments: dict[str, ArgumentSpecification] = {}
        # Map from passed argument name to option name in the struct
        # (lookup table and collision check)
        # (e.g. '--energy' -> 'energy')
        self.argumentnames: dict[str | int, str] = {}

    def add_argument(
        self,
        name: str,
        *arg_names: str,
        nargs: int | str = 1,
        type: Callable[[str], Any] = str,
        default: Any = _MISSING_ARG,
        required: bool = False,
        destination: str | None = None,
        action: str = 'store',
        const: Any = _MISSING_ARG,
    ) -> None:
        # If any arg_names start with '-', it's an option argument.
        # If so, all arg_names must start with at least one '-'.
        # Otherwise, it's a value argument.
        if all(arg_name.startswith('-') for arg_name in arg_names) and len(arg_names) > 0:
            argument_type = 'option'
        elif len(arg_names) == 1 and not arg_names[0].startswith('-'):
            argument_type = 'value'
        elif len(arg_names) == 0:
            argument_type = 'value'
            arg_names = (name,)
        else:
            raise ArgumentError(
                "All argument names must start with '-' for option arguments, or none for value arguments"
            )

        # Check that nargs is valid
        if isinstance(nargs, int) and nargs >= 0:
            pass
        elif nargs in ('*', '+', '?'):
            pass
        else:
            raise ArgumentError("nargs must be a positive integer or one of '*', '+', '?'")

        # Check if the argument name is valid (i.e. exists in the struct)
        field = next((f for f in fields(self.returnstruct) if f.name == name), None)
        if field is None:
            raise ArgumentError(f"Argument '{name}' does not exist in {self.returnstruct.__name__}")

        # Check if the argument is already defined
        if name in self.arguments:
            raise DuplicateArgumentError(f"Argument '{name}' is already defined")
        for arg_name in arg_names:
            if arg_name in self.argumentnames:
                raise DuplicateArgumentError(
                    f"Argument name '{arg_name}' is already used for argument '{self.argumentnames[arg_name]}'"
                )

        # If default is not given, use the dataclass default
        if default is _MISSING_ARG:
            if field.default is not MISSING:
                default = field.default
            elif field.default_factory is not MISSING:
                default = field.default_factory()
            else:
                default = None

        argumentaction = ActionSpecification(
            destination=destination if destination is not None else name,
            action=action,
            constant=const if const is not _MISSING_ARG else None,
        )

        if argument_type == 'value':
            position = len([v for v in self.arguments.values() if isinstance(v, ValueSpecification)])
            arg = ValueSpecification(
                name=name,
                action=argumentaction,
                nargs=nargs,
                type=type,
                required=required,
                position=position,
                default=default,
            )
            self.argumentnames[position] = name
        elif argument_type == 'option':
            arg = OptionSpecification(
                name=name,
                action=argumentaction,
                nargs=nargs,
                arg_names=list(arg_names),
                type=type,
                required=required,
                default=default,
            )
        else:
            raise RuntimeError('Unknown program state')

        self.arguments[name] = arg
        for arg_name in arg_names:
            self.argumentnames[arg_name] = name

        pass

    def parse(self, cmd: Token, tokens: Iterable[Token | Tree[Token]]) -> Command[_T]:
        linenumber = cmd.line if cmd.line else 0
        a, k = self.structinit
        output = self.returnstruct(*a, **k)
        # Set default values from the argument specifications
        for arg in self.arguments.values():
            setattr(output, arg.name, arg.default)
        for idx, token in enumerate(tokens):
            self._parse_token(output, token, idx)
        return Command(linenumber, cmd.value, output, self.metadata)

    def _parse_token(self, output: _T, token: Token | Tree[Token], index: int) -> Any:
        match token:
            case Token(_, str(value)):
                # The token is a simple value, so it must be a positional argument -> value
                try:
                    arg = self.arguments[self.argumentnames[index]]
                except KeyError:
                    raise ParseError(f"Unexpected positional argument at position {index}: '{value}'", token)
                assert isinstance(arg, ValueSpecification)

                # Parse the value
                parsed_value = arg.type(value)
                # Perform the action for the given argument
                self._perform_action(output, arg, parsed_value)

            case Tree(Token('RULE', 'option'), [Token('OPTION', str(optname)) as t, *values]):
                try:
                    opt = self.arguments[self.argumentnames[optname]]
                except KeyError:
                    raise ParseError(f"Unknown option '{optname}'", t)
                assert isinstance(opt, OptionSpecification)

                # Parse the values
                parsed_values = [opt.type(v.value) for v in values if isinstance(v, Token)]

                match opt.nargs, len(parsed_values):
                    case 1, 1:
                        self._perform_action(output, opt, parsed_values[0])
                    case int(n), m if n == m:
                        self._perform_action(output, opt, parsed_values)
                    case '*', _:
                        self._perform_action(output, opt, parsed_values)
                    case '+', m if m >= 1:
                        self._perform_action(output, opt, parsed_values)
                    case '?', m if m <= 1:
                        self._perform_action(output, opt, parsed_values[0] if m == 1 else None)
                    case int(n), m:
                        raise ParseError(
                            f"Argument '{opt.name}' expects {opt.nargs} values, but got {len(parsed_values)}", t
                        )
                    case '*', m:
                        raise ParseError(
                            f"Argument '{opt.name}' expects 0 or more values, but got {len(parsed_values)}", t
                        )
                    case '+', 0:
                        raise ParseError(f"Argument '{opt.name}' expects 1 or more values, but got 0", t)
                    case '?', m if m > 1:
                        raise ParseError(
                            f"Argument '{opt.name}' expects at most 1 value, but got {len(parsed_values)}", t
                        )
                    case _:
                        raise RuntimeError('Unknown program state')
            case Tree(Token() as t, _):
                raise ParseError('Invalid token', t)
            case _:
                raise ParseError('Invalid token')

    def _perform_action(self, output: _T, arg: ArgumentSpecification, value: Any) -> None:
        match arg.action.action:
            case 'store':
                setattr(output, arg.action.destination, value)
            case 'store_const':
                setattr(output, arg.action.destination, arg.action.constant)
            case 'set_true':
                setattr(output, arg.action.destination, True)
            case 'set_false':
                setattr(output, arg.action.destination, False)
            case 'append':
                # Assume the field is a list, as it should be created with default_factory=list
                getattr(output, arg.action.destination).append(value)
            case _:
                raise ArgumentError(f"Unknown action '{arg.action.action}' for argument '{arg.name}'")

    def validate(self, output: _T) -> None:
        # Check that all required arguments are present
        for arg in self.arguments.values():
            if getattr(output, arg.name) is _MISSING_ARG:
                raise ArgumentError(f"Argument '{arg.name}' was not provided")
            if arg.required and getattr(output, arg.name) is None:
                raise ArgumentError(f"Argument '{arg.name}' is required but not provided")

    def __call__(self, cmd: Token, args: Iterable[Token | Tree[Token]], parsecontext: ParseContext) -> Command[_T]:
        out = self.parse(cmd, args)
        self.validate(out.args)
        return out
