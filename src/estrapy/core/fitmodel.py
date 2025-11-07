import yaml
import re

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Union
from mashumaro import DataClassDictMixin


@dataclass
class Variable(DataClassDictMixin):
    name: str
    initial: Optional[float] = None
    bounds: Optional[List[float]] = None
    step: Optional[float] = None
    fix: Optional[bool] = None

@dataclass
class VariableRule(DataClassDictMixin):
    names: List[str]
    rule: str
    initial: Optional[List[float]] = None
    bounds: Optional[List[List[float]]] = None

@dataclass
class VariableCalc(DataClassDictMixin):
    name: str
    expression: str

@dataclass
class Phase(DataClassDictMixin):
    name: str
    position: Path
    N: Optional[str] = None
    r: Optional[str] = None
    dE: Optional[str] = None
    s2: Optional[str] = None
    s02: Optional[str] = None
    eta: Optional[str] = None


@dataclass
class ExafsModel(DataClassDictMixin):
    kind: str
    phasesroot: Path
    variables: List[Union[Variable,VariableRule,VariableCalc]] = field(default_factory=list[Union[Variable,VariableRule,VariableCalc]])
    phases: List[Phase] = field(default_factory=list[Phase])

def validate_model(modelpath: Path, model: ExafsModel) -> None:
    # Ensure model kind is feff, no other kinds supported yet
    if model.kind.lower() != 'feff':
        raise ValueError(f'Unsupported model kind: {model.kind}. Only "feff" is supported.')
    
    # Ensure the phases root exists
    root_path = model.phasesroot
    if not root_path.is_absolute():
        root_path = (modelpath.parent / root_path).resolve()

    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f'Phases root directory does not exist: {root_path}')
    
    # Ensure all phase paths exist
    for phase in model.phases:
        phase_path = phase.position
        if not phase_path.is_absolute():
            phase_path = (root_path / phase_path).resolve()
        
        if not phase_path.exists() or not phase_path.is_file():
            raise ValueError(f'Phase path does not exist: {phase_path}')
    
    # Ensure all variables exist
    variable_names = set[str]()
    for var in model.variables:
        match var:
            case Variable(name=name):
                variable_names.add(name)
            case VariableRule(names=names):
                variable_names.update(names)
            case VariableCalc(name=name):
                variable_names.add(name)
                # TODO: validate expression references
    
    for phase in model.phases:
        # TODO: validate expression references for
        phase.N
        phase.r
        phase.dE
        phase.s2
        phase.s02
        phase.eta
    pass

def load_model_from_yaml(path: Path, *, validate:bool=True) -> tuple[tuple[int,...], ExafsModel]:
    with open(path, "r") as f:
        first_line = f.readline()
        version_match = re.match(
            r'^#\s*version\s*:?\s*(?:(\d+)\.(\d+)(?:\.(\d+))?)\s*$',
            first_line,
            re.IGNORECASE,
        )
        if version_match is None:
            raise ValueError("Model file missing version header.")
        
        version_match = tuple(int(x) if x is not None else 0 for x in version_match.groups())

        data = yaml.safe_load(f)
    
    model = ExafsModel.from_dict(data) # type: ignore

    if validate:
        validate_model(Path(path), model)

    return version_match, model

def save_model_to_yaml(path: str, model: ExafsModel) -> None:
    yaml.safe_dump(model.to_dict(), open(path, 'w'), indent=4, encoding='utf-8', allow_unicode=True)
