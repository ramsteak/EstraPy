from typing import TypeVar, Generic, Callable, Any, Literal, Union, Protocol, Iterator
from lark import Token, Tree
from dataclasses import dataclass, fields, MISSING
from types import EllipsisType
from enum import Enum

from ..core.grammarclasses import CommandArguments
from ..core.errors import ParseError
from ..core.errors import ArgumentError, DuplicateArgumentError
from ..core.context import ParseContext
from ..core.misc import peekable

# Define a command parser that can parse commands and their arguments, akin to an argparser.
_T = TypeVar('_T', bound=CommandArguments)


class ActionType(Enum):
    STORE = 'store'
    STORE_CONST = 'store_const'
    SET_TRUE = 'set_true'
    SET_FALSE = 'set_false'
    APPEND = 'append'
    APPEND_CONST = 'append_const'


@dataclass(slots=True)
class ActionSpecification:
    destination: str
    action: ActionType
    constant: Any = None


@dataclass(slots=True, frozen=True)
class NargsSpecification:
    min: int | None
    max: int | None
    soft: bool
    single: bool

    def inrange(self, n: int) -> bool:
        if self.min is not None and n < self.min:
            return False
        if self.max is not None and n > self.max:
            return False
        return True


class Callback_var(Protocol):
    def __call__(self, *args: str) -> Any: ...


Callback = Union[
    Callable[[str], Any],
    Callable[[str, str], Any],
    Callable[[str, str, str], Any],
    Callable[[str, str, str, str], Any],
    Callable[[str, str, str, str, str], Any],
    Callable[[str, str, str, str, str, str], Any],
    Callable[[str, str, str, str, str, str, str], Any],
    Callback_var,
]


@dataclass(slots=True)
class ArgumentSpecification:
    name: str
    action: ActionSpecification
    type: Callable[[str], Any] = str
    types: Callback | None = None  # For multiple types # type: ignore
    required: bool = False
    default: Any = None
    nargs: NargsSpecification = NargsSpecification(1, 1, False, True)
    accept: tuple[str, ...] | None = None
    default_factory: Callable[[], Any] | None = None


def _validate_nargs(
    nargs: int | str | tuple[int | None | EllipsisType, int | None | EllipsisType],
) -> NargsSpecification:
    """Validate the nargs parameter. Raises ArgumentError if invalid."""
    match nargs:
        case int(n) if n >= 0:
            return NargsSpecification(min=n, max=n, soft=False, single=(n == 1))
        case [int(minn), int(maxn)] if minn >= 0 and maxn >= minn:
            return NargsSpecification(min=minn, max=maxn, soft=False, single=False)
        case [None, int(maxn)] if maxn >= 0:
            return NargsSpecification(min=0, max=maxn, soft=False, single=False)
        case [EllipsisType(), int(maxn)] if maxn >= 0:
            return NargsSpecification(min=0, max=maxn, soft=True, single=False)
        case [int(minn), None] if minn >= 0:
            return NargsSpecification(min=minn, max=None, soft=False, single=False)
        case [int(minn), EllipsisType()] if minn >= 0:
            return NargsSpecification(min=minn, max=None, soft=True, single=False)
        case '*' | [0, None] | [None, None]:
            return NargsSpecification(min=0, max=None, soft=False, single=False)
        case '+' | [1, None]:
            return NargsSpecification(min=1, max=None, soft=False, single=False)
        case '?' | [None, 1]:
            return NargsSpecification(min=0, max=1, soft=False, single=True)
        case '*?' | [0, EllipsisType()] | [EllipsisType(), EllipsisType()]:
            return NargsSpecification(min=0, max=None, soft=True, single=False)
        case '+?' | [1, EllipsisType()]:
            return NargsSpecification(min=1, max=None, soft=True, single=False)
        case '??' | [EllipsisType(), 1]:
            return NargsSpecification(min=0, max=1, soft=True, single=True)
        case _:
            raise ArgumentError(
                f"nargs must be a positive integer, one of '*', '+', '?', or a tuple of two positive integers (min, max) with min <= max, not '{nargs}'"
            )


def _get_argtype_from_names(arg_names: tuple[str, ...]) -> Literal['option', 'value']:
    """Determine if the argument is an option or value based on its names.
    Raises ArgumentError if the names are invalid."""
    if all(name.startswith('-') for name in arg_names) and len(arg_names) > 0:
        return 'option'
    elif len(arg_names) == 0:
        return 'value'
    else:
        raise ArgumentError(
            "All argument names must start with '-' for option arguments. No argument names for value arguments are required."
        )


def _get_action_specification(destination: str, action: ActionType | str, const: Any) -> ActionSpecification:
    """Get the ActionSpecification for the given argument parameters."""
    if isinstance(action, str):
        try:
            action = ActionType(action)
        except ValueError:
            raise ArgumentError(f"Unknown action type '{action}' for argument '{destination}'.")

    match action:
        case ActionType.STORE:
            return ActionSpecification(
                destination=destination,
                action=ActionType.STORE,
                constant=None,
            )
        case ActionType.STORE_CONST:
            if const is MISSING:
                raise ArgumentError('Constant value must be provided for STORE_CONST action.')
            return ActionSpecification(
                destination=destination,
                action=ActionType.STORE_CONST,
                constant=const,
            )
        case ActionType.SET_TRUE:
            return ActionSpecification(
                destination=destination,
                action=ActionType.STORE_CONST,
                constant=True,
            )
        case ActionType.SET_FALSE:
            return ActionSpecification(
                destination=destination,
                action=ActionType.STORE_CONST,
                constant=False,
            )
        case ActionType.APPEND:
            return ActionSpecification(
                destination=destination,
                action=ActionType.APPEND,
                constant=None,
            )
        case ActionType.APPEND_CONST:
            if const is MISSING:
                raise ArgumentError('Constant value must be provided for APPEND_CONST action.')
            return ActionSpecification(
                destination=destination,
                action=ActionType.APPEND_CONST,
                constant=const,
            )
        case _:
            raise ArgumentError(f"Unknown action type '{action}' for argument '{destination}'.")


class CommandArgumentParser(Generic[_T]):
    def __init__(self, returnstruct: type[_T], name: str | None = None) -> None:
        # Check that returnstruct is a dataclass
        if not hasattr(returnstruct, '__dataclass_fields__'):
            raise TypeError('returnstruct must be a dataclass')

        self.name = name
        self.dataclass_type = returnstruct
        self._fields = {f.name: f for f in fields(returnstruct)}

        # Return type argument name -> ArgumentSpecification
        # Unnamed arguments are stored with a progressive index as key
        # (e.g. store_const arguments, requires destination)
        self.arguments: dict[str, ArgumentSpecification] = {}
        self._unnamed_arg_index = 0
        # Passed argument name (or position for positional args) -> return type argument name
        # This is used for lookup at parse time and duplicate detection at instantiation time
        self.command_posargs: list[str] = []
        self.command_optflags: dict[str, str] = {}

        # Subparsers of the main command parser
        self.subparsers: dict[str, CommandArgumentParser[Any]] = {}  # name -> parser
        self.subparser_destinations: dict[str, str] = {}  # name -> destination in returnstruct

        self.parent: None | CommandArgumentParser[Any] = None

    def add_argument(
        self,
        field: str | None,
        *flags: str,
        nargs: int | Literal['*', '*?', '?', '??', '+', '+?'] | tuple[int, int] = 1,
        type: Callable[[str], Any] = str,
        types: Callback | None = None,
        default: Any = MISSING,  # type: ignore
        default_factory: Callable[[], Any] = MISSING,
        required: bool = False,
        dest: str | None = None,  # type: ignore
        action: ActionType | str = ActionType.STORE,
        const: Any = MISSING,
        accept: str | tuple[str, ...] | None = None,
    ) -> None:
        # ------------------------------------------------------------------------------------------------------
        # Resolve destination and name
        if field is None:
            if dest is None:
                raise ArgumentError('Destination must be given for unnamed arguments.')
            # Assign a progressive attr name when not given.
            self._unnamed_arg_index += 1
            field = f'unnamed_arg_{self._unnamed_arg_index}'
        else:
            # If field is given, destination defaults to field, and cannot be given.
            if dest is not None and dest != field:
                raise ArgumentError('Destination cannot be given when field is specified.')
            dest = field
            # If field is given, check that it exists in the dataclass_type and that it's not already defined.
            if field not in self._fields:
                raise ArgumentError(f"Argument '{field}' does not exist in {self.dataclass_type.__name__}.")
            if field in self.arguments:
                raise DuplicateArgumentError(f"Argument '{field}' is already defined.")

        # ------------------------------------------------------------------------------------------------------
        # Validate nargs and defaults
        _nargs = _validate_nargs(nargs)

        if default is not MISSING and default_factory is not MISSING:
            raise ArgumentError('Cannot specify both default and default_factory.')

        # ------------------------------------------------------------------------------------------------------
        # Check for duplicate flag entries
        for name in flags:
            if not name.startswith('-'):
                raise ArgumentError(f"Invalid flag name '{name}': option flags must start with '-'")
            if name in self.command_optflags:
                raise DuplicateArgumentError(
                    f"Flag name '{name}' is already used for argument '{self.command_optflags[name]}'"
                )

        # ------------------------------------------------------------------------------------------------------
        # Build ArgumentSpecification for this argument
        action_spec = _get_action_specification(dest, action, const)

        if action_spec.action in {ActionType.STORE, ActionType.APPEND} and nargs == 0:
            raise ArgumentError(f"Argument '{field}': nargs cannot be 0 for STORE or APPEND actions.")
        if action_spec.action in {ActionType.STORE_CONST, ActionType.APPEND_CONST} and nargs != 0:
            raise ArgumentError(f"Argument '{field}': nargs must be 0 for STORE_CONST or APPEND_CONST actions.")

        argument_spec = ArgumentSpecification(
            name=field,
            action=action_spec,
            type=type,
            types=types,
            required=required,
            default=default,
            nargs=_nargs,
            default_factory=default_factory,
            accept=accept if isinstance(accept, tuple) or accept is None else (accept,),
        )

        # ------------------------------------------------------------------------------------------------------
        # Register argument
        self.arguments[field] = argument_spec

        match _get_argtype_from_names(flags):
            case 'value':
                self.command_posargs.append(field)
            case 'option':
                opt_handles = flags
                for handle in opt_handles:
                    self.command_optflags[handle] = field
            case _:
                raise RuntimeError('Unknown argument type')

    def add_subparser(self, name: str, parser: 'CommandArgumentParser[Any]', dest: str) -> None:
        # ------------------------------------------------------------------------------------------------------
        # Validate destination and flags
        if dest not in self._fields:
            raise ArgumentError(f"Argument '{dest}' does not exist in {self.dataclass_type.__name__}.")
        if dest in self.arguments and dest not in self.subparser_destinations.values():
            raise DuplicateArgumentError(f"Argument '{dest}' is already defined.")
        if name in self.subparsers:
            raise DuplicateArgumentError(f"Subparser '{name}' is already defined.")

        # ------------------------------------------------------------------------------------------------------
        # Register subparser
        self.subparsers[name] = parser
        self.subparser_destinations[name] = dest

        # Link hierarchy (optional, for debugging or nested parsing)
        parser.parent = self
        parser.name = parser.name or name

        # TODO: add to arguments
        action_spec = _get_action_specification(dest, ActionType.STORE, None)
        argument_spec = ArgumentSpecification(
            name=dest,
            action=action_spec,
            type=lambda x: x,  # type: ignore
            required=True,
        )
        self.arguments[dest] = argument_spec

    def parse(self, commandtoken: Token, tokens: peekable[Token | Tree[Token]]) -> _T:
        kwargs: dict[str, Any] = {}

        # --------------------------------------------------------------------------------------
        # Get default values from dataclass fields
        for fieldname, field in self._fields.items():
            if field.default is not MISSING:
                kwargs[fieldname] = field.default
            elif field.default_factory is not MISSING:
                kwargs[field.name] = field.default_factory()

        # Get default values from ArgumentSpecifications
        for argname, argspec in self.arguments.items():
            if argspec.default is not MISSING:
                kwargs[argname] = argspec.default
            elif argspec.default_factory is not MISSING and argspec.default_factory is not None:
                kwargs[argname] = argspec.default_factory()

        # --------------------------------------------------------------------------------------
        # Main parsing loop
        _allfields = set(self._fields.keys())
        _definedfields: set[str] = set()
        _positional_arguments_iter = (self.arguments[name] for name in self.command_posargs)

        while tokens and (_allfields - _definedfields):
            # Get first token for error reporting
            __t = tokens.peek()
            while isinstance(__t, Tree):
                __t = __t.children[0]

            # Attempt to parse first token as subparser
            result = self._parse_subparser(tokens)
            if result is not None:
                arg, val = result
                if arg.action.destination in _definedfields:
                    raise ParseError(f"Argument '{arg.action.destination}' is already defined.", __t)
                self._perform_action(kwargs, arg.action, val)
                _definedfields.add(arg.action.destination)
                continue

            # Attempt to parse first token as option
            result = self._parse_option(tokens)
            if result is not None:
                arg, val = result
                if arg.action.destination in _definedfields:
                    raise ParseError(f"Argument '{arg.action.destination}' is already defined.", __t)
                self._perform_action(kwargs, arg.action, val)
                _definedfields.add(arg.action.destination)
                continue

            # Attempt to parse first token as a positional argument
            result = self._parse_positional(tokens, _positional_arguments_iter)
            if result is not None:
                arg, val = result
                if arg.action.destination in _definedfields:
                    raise ParseError(f"Argument '{arg.action.destination}' is already defined.", __t)
                self._perform_action(kwargs, arg.action, val)
                _definedfields.add(arg.action.destination)
                continue

            # None matched, so this argument parser is finished and we exit
            break

        for argname, argspec in self.arguments.items():
            if argspec.required and argname not in kwargs:
                raise ArgumentError(f"Argument '{argname}' is required but not provided.")

        output = self.dataclass_type(**kwargs)
        return output

    def _parse_subparser(self, tokens: peekable[Token | Tree[Token]]) -> tuple[ArgumentSpecification, Any] | None:
        if not self.subparsers:
            return None

        next_token = next(tokens, None)
        if next_token is None:
            return None

        # Subparser is triggered by a positional token matching a subparser name
        if not isinstance(next_token, Token):
            tokens.pushback(next_token)
            return None

        if next_token.value not in self.subparsers:
            tokens.pushback(next_token)
            return None

        # --------------------------------------------------------------------------------------
        # Parse subparser
        subp = self.subparsers[next_token.value]
        dest = self.subparser_destinations[next_token.value]

        parsed = subp.parse(next_token, tokens)

        return self.arguments[dest], parsed

    def _parse_option(self, tokens: peekable[Token | Tree[Token]]) -> tuple[ArgumentSpecification, Any] | None:
        __t = tokens.peek()
        while isinstance(__t, Tree):
            __t = __t.children[0]

        # Option is triggered by a Tree with Tree(Token('RULE', 'option'), [...])
        # Match used to destructure and refuse invalid tokens
        token = next(tokens, None)
        match token:
            case None:
                return None
            case Tree(Token('RULE', 'option'), [Token('OPTION', str(option)) as _opt, *subtokens]):
                arg_name = self.command_optflags.get(option)
                if arg_name is None:
                    tokens.pushback(token)
                    return None
                arg_spec = self.arguments[arg_name]

                # Handle STORE_CONST / APPEND_CONST directly
                if arg_spec.action.action in (ActionType.STORE_CONST, ActionType.APPEND_CONST):
                    tokens.pushback_n(subtokens)
                    return arg_spec, arg_spec.action.constant

                # Parse values according to nargs
                sub_tokens = peekable(subtokens)
                value = self._consume_tokens_for_nargs(sub_tokens, arg_spec.nargs, arg_spec.type, arg_spec.accept)
                if arg_spec.types is not None and not arg_spec.nargs.single:
                    try:
                        value = arg_spec.types(*value)
                    except Exception as e:
                        tokens.pushback(token)
                        raise ParseError(f"Error parsing argument '{arg_spec.name}': {e}", _opt) from e
                # Push back all unconsumed tokens
                tokens.pushback_n(sub_tokens)
                return arg_spec, value

            case _:
                tokens.pushback(token)
                return None

        ...

    def _parse_positional(
        self, tokens: peekable[Token | Tree[Token]], posargs: Iterator[ArgumentSpecification]
    ) -> tuple[ArgumentSpecification, Any] | None:
        arg_spec = next(posargs, None)
        if arg_spec is None:
            return None
        __t = tokens.peek()
        while isinstance(__t, Tree):
            __t = __t.children[0]

        try:
            value = self._consume_tokens_for_nargs(tokens, arg_spec.nargs, arg_spec.type, arg_spec.accept)
        except ParseError as pe:
            if not arg_spec.required:
                return None
            raise pe from None

        if arg_spec.types is not None and not arg_spec.nargs.single:
            try:
                value = arg_spec.types(*value)
            except Exception as e:
                raise ParseError(f"Error parsing argument '{arg_spec.name}': {e}", __t) from e

        return arg_spec, value

    def _consume_tokens_for_nargs(
        self,
        tokens: peekable[Token | Tree[Token]],
        nargs_spec: NargsSpecification,
        _type: Callable[[str], Any],
        accept: tuple[str, ...] | None = None,
    ) -> list[Any] | Any:
        values: list[Any] = []
        __tokens: list[Token] = []
        __t = tokens.peek()
        while isinstance(__t, Tree):
            __t = __t.children[0]

        while tokens:
            token = tokens.peek()
            if not isinstance(token, Token):
                break  # Cannot consume non-value token here

            # If accept is defined, stop at first non-accepted token
            if accept is not None and token.type not in accept:
                break

            try:
                val = _type(token.value)
            except Exception:
                # Soft mode: stop consuming and leave token on stack
                if nargs_spec.soft:
                    break
                else:
                    tokens.pushback_n(__tokens)
                    raise ParseError(f"Failed to parse token '{token.value}' as {_type}", token) from None

            values.append(val)
            __tokens.append(tokens.next())  # type: ignore

            if nargs_spec.max is not None and len(values) >= nargs_spec.max:
                break

        # Validate number of consumed tokens
        if not nargs_spec.inrange(len(values)):
            tokens.pushback_n(__tokens)
            raise ParseError(
                f'Expected between {nargs_spec.min} and {nargs_spec.max} values, but got {len(values)}', __t
            )

        if nargs_spec.single:
            return values[0] if values else None
        return values

    def _perform_action(self, kwargs: dict[str, Any], action: ActionSpecification, value: Any) -> None:
        match action.action:
            case ActionType.STORE:
                kwargs[action.destination] = value
            case ActionType.STORE_CONST:
                kwargs[action.destination] = action.constant
            case ActionType.APPEND:
                kwargs.setdefault(action.destination, []).append(value)
            case ActionType.APPEND_CONST:
                kwargs.setdefault(action.destination, []).append(action.constant)
            case _:
                raise ArgumentError(f"Unknown action '{action.action}' for argument '{action.destination}'")

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def __call__(self, commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext) -> _T:
        _tokens = peekable(tokens)
        struct = self.parse(commandtoken, _tokens)

        if not self.is_root:
            return struct

        if _tokens:
            match _tokens.next():
                case Token() as t:
                    raise ParseError('Unexpected extra tokens after parsing command.', t)
                case Tree(Token() as t, _):
                    raise ParseError('Unexpected extra tokens after parsing command.', t)
                case _:
                    raise ParseError('Unexpected extra tokens after parsing command.')

        _tokens.close()
        return struct
