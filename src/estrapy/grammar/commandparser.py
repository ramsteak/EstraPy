from typing import TypeVar, Generic, Callable, Any, Literal
from lark import Token, Tree
from dataclasses import dataclass, fields, MISSING
from enum import Enum

from ..core.grammarclasses import CommandArguments
from ..core.errors import ParseError
from ..core.errors import ArgumentError, DuplicateArgumentError
from ..core.context import ParseContext

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


@dataclass(slots=True)
class ArgumentSpecification:
    name: str
    action: ActionSpecification
    type: Callable[[str], Any] = str
    required: bool = False
    default: Any = None
    nargs: int | str | tuple[int, int] = 1
    accept: tuple[str, ...] | None = None
    default_factory: Callable[[], Any] | None = None



def _get_dataclass_defaults(cls: type[Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for field in fields(cls):
        if field.default is not MISSING:
            defaults[field.name] = field.default
        elif field.default_factory is not MISSING:
            defaults[field.name] = field.default_factory()
    return defaults

def _validate_nargs(nargs: int | str | tuple[Any,...]) -> None:
    """Validate the nargs parameter. Raises ArgumentError if invalid."""
    # Valid nargs are:
    #  - positive integer
    #  - '*', '+', '?'
    #  - tuple of two positive integers (min, max), inclusive, with min <= max
    match nargs:
        case int(n) if n >= 0:
            return
        case '*' | '+' | '?' | '*?' | '??' | '+?':
            return
        case [int(minn), int(maxn)] if minn >= 0 and maxn >= minn:
            return
        case _:
            raise ArgumentError(f"nargs must be a positive integer, one of '*', '+', '?', or a tuple of two positive integers (min, max) with min <= max, not '{nargs}'")

def _get_argtype_from_names(arg_names: tuple[str, ...]) -> Literal['option', 'value']:
    """Determine if the argument is an option or value based on its names.
    Raises ArgumentError if the names are invalid."""
    if all(name.startswith('-') for name in arg_names) and len(arg_names) > 0:
        return 'option'
    elif len(arg_names) == 0:
        return 'value'
    else:
        raise ArgumentError("All argument names must start with '-' for option arguments. No argument names for value arguments are required.")

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
                raise ArgumentError("Constant value must be provided for STORE_CONST action.")
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
                raise ArgumentError("Constant value must be provided for APPEND_CONST action.")
            return ActionSpecification(
                destination=destination,
                action=ActionType.APPEND_CONST,
                constant=const,
            )
        case _:
            raise ArgumentError(f"Unknown action type '{action}' for argument '{destination}'.")

class CommandArgumentParser(Generic[_T]):
    def __init__(self, returnstruct: type[_T]) -> None:
        # Check that returnstruct is a dataclass
        if not hasattr(returnstruct, '__dataclass_fields__'):
            raise TypeError('returnstruct must be a dataclass')
        
        self.command_returnstruct = returnstruct
        self._returnstruct_fields = {f.name:f for f in fields(returnstruct)}

        # Return type argument name -> ArgumentSpecification
        # Unnamed arguments are stored with a progressive index as key
        # (e.g. store_const arguments, requires destination)
        self.command_arguments: dict[str, ArgumentSpecification] = {}
        self._unnamed_arg_index = 0
        self._positional_arg_index = 0
        # Passed argument name (or position for positional args) -> return type argument name
        # This is used for lookup at parse time and duplicate detection at instantiation time
        self.command_posargs: list[str] = []
        self.command_optflags: dict[str, str] = {}
    
    def add_argument(self, attr_name: str | None, *arg_name: str, nargs: int | Literal['*', '*?', '?', '??', '+', '+?'] | tuple[int,int] = 1,
                     type: Callable[[str], Any] = str, default: Any = MISSING, default_factory: Callable[[], Any] = MISSING, # type: ignore
                     required: bool = False, destination: str | None = None,
                     action: ActionType | str = ActionType.STORE, const: Any = MISSING, accept: str | tuple[str, ...] | None = None) -> None:
    
        if attr_name is None:
            if destination is None:
                # If not given, destination is required.
                raise ArgumentError("Destination must be given for unnamed arguments.")
            # Assign a progressive attr name when not given.
            self._unnamed_arg_index += 1
            attr_name = f'unnamed_arg_{self._unnamed_arg_index}'
        else:
            # If attr_name is given, destination defaults to attr_name, and cannot be given.
            if destination is not None and destination != attr_name:
                raise ArgumentError("Destination cannot be given when attr_name is specified.")
            destination = attr_name
            # If attr_name is given, check that it exists in the returnstruct and that it's not already defined.
            if attr_name not in self._returnstruct_fields:
                raise ArgumentError(f"Argument '{attr_name}' does not exist in {self.command_returnstruct.__name__}.")
            if attr_name in self.command_arguments:
                raise DuplicateArgumentError(f"Argument '{attr_name}' is already defined.")
        
        # Validate the nargs parameter
        _validate_nargs(nargs)

        # Check for duplicate arg_name entries
        for name in [a for a in arg_name if a in self.command_optflags]:
            raise DuplicateArgumentError(f"Argument name '{name}' is already used for argument '{self.command_optflags[name]}'")

        # Get the action for the given argument
        argument_action = _get_action_specification(destination, action, const)

        if argument_action.action in {ActionType.STORE, ActionType.APPEND} and nargs == 0:
            raise ArgumentError(f"Argument '{attr_name}': nargs cannot be 0 for STORE or APPEND actions.")
        if argument_action.action in {ActionType.STORE_CONST, ActionType.APPEND_CONST} and nargs != 0:
            raise ArgumentError(f"Argument '{attr_name}': nargs must be 0 for STORE_CONST or APPEND_CONST actions.")
        
        # Check that default and default_factory are not both given
        if default is not MISSING and default_factory is not MISSING:
            raise ArgumentError("Cannot specify both default and default_factory.")
        
        argument_spec = ArgumentSpecification(
            name=attr_name,
            action=argument_action,
            type=type,
            required=required,
            default=default,
            nargs=nargs,
            default_factory=default_factory,
            accept=accept if isinstance(accept, tuple) or accept is None else (accept,),
        )

        self.command_arguments[attr_name] = argument_spec

        match _get_argtype_from_names(arg_name):
            case 'value':
                self.command_posargs.append(attr_name)
            case 'option':
                opt_handles = arg_name
                for handle in opt_handles:
                    self.command_optflags[handle] = attr_name
            case _: 
                raise RuntimeError('Unknown program state')
    

    def parse(self, commandtoken: Token, tokens: list[Token | Tree[Token]]) -> _T:
        # Get default values from the dataclass, both from the dataclass itself
        # and from the ArgumentSpecifications. ArgumentSpecifications take precedence.
        defaults = _get_dataclass_defaults(self.command_returnstruct)
        # Get default values from ArgumentSpecifications
        for argname, argspec in self.command_arguments.items():
            if argspec.default is not MISSING:
                defaults[argname] = argspec.default
            elif argspec.default_factory is not MISSING and argspec.default_factory is not None:
                defaults[argname] = argspec.default_factory()

        # Split positional and option tokens. Positional tokens always come first.
        positional_tokens: list[Token] = []
        option_tokens: list[Tree[Token]] = []
        for token in tokens:
            match token:
                case Token(_, _):
                    positional_tokens.append(token)
                case Tree(Token('RULE', 'option'), _):
                    option_tokens.append(token)
                case Tree(Token() as t, _):
                    raise ParseError('Invalid token', t)
                case _:
                    raise ParseError('Invalid token')

        # Parse positional tokens
        for arg, value in self._parse_positionals(positional_tokens):
            self._perform_action(defaults, arg.action, value)


        # Parse option tokens
        for option_token in option_tokens:
            arg, value = self._parse_option(option_token)
            self._perform_action(defaults, arg.action, value)


        for argname, argspec in self.command_arguments.items():
            if argspec.required and argname not in defaults:
                raise ArgumentError(f"Argument '{argname}' is required but not provided.")
        
        output = self.command_returnstruct(**defaults)

        return output
    
    def _parse_positionals(self, tokens: list[Token]) -> list[tuple[ArgumentSpecification, Any]]:
        arguments: list[tuple[ArgumentSpecification, Any]] = []

        # This is so that we can get the next token if nargs > 1
        _iter_idx_tokens = iter(enumerate(tokens))
        # Represents the current positional argument index
        posargs = (self.command_arguments[name] for name in self.command_posargs)
        for arg in posargs:
            values:list[Any] = []

            match arg.nargs:
                case int(n):
                    for _ in range(n):
                        token = None
                        try:
                            _, token = next(_iter_idx_tokens)
                            values.append(arg.type(token.value))
                        except StopIteration:
                            raise ParseError(f"Argument '{arg.name}' expects {n} values, but got {len(values)}", tokens[-1])
                        except Exception as e:
                            raise ParseError(f"Error parsing argument '{arg.name}': {e}", token)
                case '*':
                    for _, token in _iter_idx_tokens:
                        values.append(arg.type(token.value))
                case '+':
                    for _, token in _iter_idx_tokens:
                        values.append(arg.type(token.value))
                    if len(values) == 0:
                        raise ParseError(f"Argument '{arg.name}' expects 1 or more values, but got 0", tokens[-1])
                case '?':
                    token = None
                    try:
                        _, token = next(_iter_idx_tokens)
                        values.append(arg.type(token.value))
                    except StopIteration:
                        pass
                    except Exception as e:
                        raise ParseError(f"Error parsing argument '{arg.name}': {e}", token)
                case '*?': # zero or more, release if error
                    for _, token in _iter_idx_tokens:
                        try:
                            values.append(arg.type(token.value))
                        except Exception:
                            break
                case '+?': # one or more, release if error
                    token = None
                    for _, token in _iter_idx_tokens:
                        try:
                            values.append(arg.type(token.value))
                        except Exception:
                            break
                    if len(values) == 0:
                        raise ParseError(f"Argument '{arg.name}' expects 1 or more values, but got 0", tokens[-1])
                case '??': # zero or one, release if error
                    token = None
                    try:
                        _, token = next(_iter_idx_tokens)
                        values.append(arg.type(token.value))
                    except StopIteration:
                        pass
                    except Exception:
                        pass
                case [int(minn), int(maxn)]:
                    for _ in range(maxn):
                        token = None
                        try:
                            _, token = next(_iter_idx_tokens)
                            values.append(arg.type(token.value))
                        except StopIteration:
                            break
                        except Exception as e:
                            if len(values) < minn:
                                raise ParseError(f"Error parsing argument '{arg.name}': {e}", token)
                            break
                    if not (minn <= len(values) <= maxn):
                        raise ParseError(f"Argument '{arg.name}' expects between {minn} and {maxn} values, but got {len(values)}", tokens[-1])
                case _:
                    raise RuntimeError('Unknown program state')        

            if arg.nargs == 1:
                arguments.append((arg, values[0]))
            elif arg.nargs in ('?', '??') and len(values) == 0:
                arguments.append((arg, None))
            elif arg.nargs in ('?', '??') and len(values) == 1:
                arguments.append((arg, values[0]))
            else:
                arguments.append((arg, values))
        return arguments

    def _parse_option(self, token: Tree[Token]) -> tuple[ArgumentSpecification, Any]:
        match token:
            # case Token(tokenkind, str(value)): # positional argument. Check that index is in command_argnames
            #     try:
            #         arg = self.command_arguments[self.command_argnames[index]]
            #     except KeyError:
            #         raise ParseError(f"Unexpected positional argument at position {position}: '{value}'", token)
                
            #     if arg.accept is not None and tokenkind not in arg.accept:
            #         raise ParseError(f"Argument '{arg.name}' does not accept token of type '{tokenkind}' '{value}'", token)
                
            #     return arg, value

            case Tree(Token('RULE', 'option'), [Token('OPTION', str(optname)) as t, *values]):
                try:
                    arg = self.command_arguments[self.command_optflags[optname]]
                except KeyError:
                    raise ParseError(f"Unknown option '{optname}'", t)
                
                if arg.accept is not None:
                    for v in values:
                        if isinstance(v, Token) and v.type not in arg.accept:
                            raise ParseError(f"Argument '{arg.name}' does not accept token of type '{v.type}' '{v.value}'", v)
                
                parsed_values = [arg.type(v.value) for v in values if isinstance(v, Token)]

                match arg.nargs, len(parsed_values):
                    case 1, 1:
                        return arg, parsed_values[0]
                    case int(n), m if n == m:
                        return arg, parsed_values
                    case [int(minn), int(maxn)], m if minn <= m <= maxn:
                        return arg, parsed_values
                    case '*', _:
                        return arg, parsed_values
                    case '+', m if m >= 1:
                        return arg, parsed_values
                    case '?', m if m <= 1:
                        return arg, parsed_values[0] if m == 1 else None
                    case int(n), m:
                        raise ParseError(f"Argument '{arg.name}' expects {arg.nargs} values, but got {len(parsed_values)}", t)
                    case [int(minn), int(maxn)], m:
                        raise ParseError(f"Argument '{arg.name}' expects between {minn} and {maxn} values, but got {len(parsed_values)}", t)
                    case '+', 0:
                        raise ParseError(f"Argument '{arg.name}' expects 1 or more values, but got 0", t)
                    case '?', m if m > 1:
                        raise ParseError(f"Argument '{arg.name}' expects at most 1 value, but got {len(parsed_values)}", t)
                    case _:
                        raise RuntimeError('Unknown program state')
            case _:
                raise ParseError('Invalid token')
                
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

    def __call__(self, commandtoken: Token, tokens: list[Token | Tree[Token]], parsecontext: ParseContext) -> _T:
        return self.parse(commandtoken, tokens)
