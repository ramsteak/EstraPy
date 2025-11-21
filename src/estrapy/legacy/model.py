
import yaml
import re

from typing import Any
from pathlib import Path

from ..core.fit.fitmodel import VariableSpec, PhaseSpec, ExafsModelSpecification
from ..core.misc import peekable

re_varline = re.compile(r"^\s*(\d+)\'\s*(\w+)\s*\'\s*([\d\.\-e\+]+)\s+([\d\.\-e\+]+)(?:\s+([\d\.\-e\+]+)\s+([\d\.\-e\+]+))?\s*$")

def remove_comment(line: str) -> str:
    """Remove comments from a line, denoted by a # character."""
    return line.split('#', 1)[0].strip()

def remove_quotes(s: str) -> str:
    """Remove surrounding quotes from a string, if present."""
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

def _parse_variable_line(line: str) -> tuple[int, str, float, float, float | None, float | None]:
    match = re_varline.match(line)
    if not match:
        raise ValueError(f"Invalid variable line format: {line}")
    varnum = int(match.group(1))
    varname = match.group(2)
    init = float(match.group(3))
    step = float(match.group(4))
    lower = float(match.group(5)) if match.group(5) is not None else None
    upper = float(match.group(6)) if match.group(6) is not None else None
    return varnum, varname, init, step, lower, upper

def _parse_variable_block(lines: peekable[str]) -> list[tuple[int, str, float, float, float | None, float | None]]:
    variables: list[tuple[int, str, float, float, float | None, float | None]] = []
    for line in lines:
        if line.startswith('*'):
            continue
        line = line.strip()
        if line == '':
            break
        variable = _parse_variable_line(line)
        variables.append(variable)
    return variables

def _parse_data_block(lines: peekable[str]) -> dict[str, Any]:
    ls = [remove_comment(l) for l in lines.next_n(7)]
    if ls[-1].strip() != '':
        raise ValueError("Data block not properly terminated with a blank line.")
    
    return {
        'filename': remove_quotes(ls[0]),
        'extension': ls[1],
        'kcol': int(ls[2].split(',')[0]),
        'kchicol': int(ls[2].split(',')[1]),
        'krange': (float(ls[3].split(',')[0]), float(ls[3].split(',')[1])),
        'kweight': float(ls[4]),
        's02_idx': int(ls[5].split(',')[0]),
        'eta_idx': int(ls[5].split(',')[1]),
    }

def _parse_expression_or_variables(line: str) -> str | tuple[int, float] | int:
    line = line.strip()
    if line.startswith('&'):
        expr = line[1:].split('&', 1)[0].strip()
        return expr
    else:
        line = remove_comment(line)
        if ',' in line:
            varidx, val = line.split(',', 1)
            return int(varidx), float(val)
        else:
            return int(line)

def _parse_fitp_block(lines: peekable[str]) -> dict[str, Any]:
    num_phases = int(remove_comment(next(lines)))
    lines.next()  # ignore line

    pars = lines.next().split()  # kmin kmax kweight shelltype sig maxR dR RL RR

    if lines.next().strip() != '':
        raise ValueError("Fit parameters block not properly terminated with a blank line.")

    return {
        'num_phases': num_phases,
        'kmin': float(pars[0]) if pars[0] != '0' else None,
        'kmax': float(pars[1]) if pars[1] != '0' else None,
        'kweight': float(pars[2]) if pars[2] != '0' else None,
        'shelltype': {'1':"H", '2':"G"}.get(pars[3]),
        'sigma': float(pars[4]),
        'Rup': float(pars[5]),
        'dR': float(pars[6]),
        'RL': float(pars[7]),
        'RR': float(pars[8]),
    }

def _parse_single_phase_block(lines: peekable[str]) -> dict[str, Any]:
    ls = [lines.next().strip() for _ in range(8)]
    shelltype = remove_comment(ls[0])
    multiplicity = _parse_expression_or_variables(ls[1])
    distance = _parse_expression_or_variables(ls[2])
    sigma2 = int(remove_comment(ls[3]))
    deltae, gamma = map(int, remove_comment(ls[4]).split(','))
    if remove_comment(ls[5]) != 'feff':
        raise ValueError("Expected 'feff' line in phase block.")
    feff_path = remove_quotes(ls[6])

    if remove_comment(ls[-1]) != '':
        raise ValueError("Phase block not properly terminated with a blank line.")
    
    return {
        'shelltype': shelltype,
        'multiplicity': multiplicity,
        'distance': distance,
        'sigma2': sigma2,
        'deltae': deltae,
        'gamma': gamma,
        'feff_path': feff_path,
    }

def _parse_all_phases_block(lines: peekable[str], numphases: int) -> list[dict[str, Any]]:
    phases: list[dict[str, Any]] = []
    for _ in range(numphases):
        phase = _parse_single_phase_block(lines)
        phases.append(phase)
    return phases

def _resolve_var_reference(ref: str | int | tuple[int, float], variables: dict[int, VariableSpec]) -> str:
    """Resolve variable reference (or calculation) to variable name or expression string."""
    match ref:
        case str(calc):
            def mapper(m: re.Match[str]) -> str:
                varid = int(m.group(1))
                if varid not in variables:
                    raise ValueError(f"Variable #{varid} not found for calculation.")
                return variables[varid].name
            
            return re.sub(r'#(\d+)', mapper, calc)
        case int(varid):
            if varid not in variables:
                raise ValueError(f"Variable #{varid} not found.")
            return variables[varid].name
        case [int(varid), float(val)]:
            if varid not in variables:
                raise ValueError(f"Variable #{varid} not found.")
            varname = variables[varid].name
            return f"({varname} * {val})"
        case _:
            raise ValueError(f"Invalid variable reference: {ref}")

def load_model_from_legacy(path: Path) -> ExafsModelSpecification:
    # The file follows a fixed rigid structure (in this comment, lines starting with ! are descriptive):
    # *** comment line (maybe optional)
    #! start of the variable block, can be variable in length
    # varnum' varname  '   initial  step  lowerbound upperbound    # fix is inferred by step, bounds are optional
    #! a blank line denotes the end of the variable block
    # blank line
    #! start of the data block, of fixed structure
    # file name (possibly with quotes around it)
    # four character identifier for the calculation (usually dum_)
    # index of the k column (1-based), index of the k*chi column (1-based)
    # kmin,kmax
    # k weight for the fit
    # variable index for s02, variable index for eta (if 0, eta is set to 3)
    #! end of the data block denoted by a blank line
    # blank line
    #! start of the phases block, the number of phases is defined, and the structure is rigid
    # number of phases
    #! one ignored line
    # kmin  kmax  kweight  shelltype  sig  maxR  dR  RL  RR
    #! blank line
    #! now the phases are defined, one block (denoted by a blank line) per phase
    # shell type (G for gaussian, H for ??)
    # multiplicity. (can be either a variable index or a calculation. If a calculation, it is surrounded by &  & and the variable id is preceded by #)
    # distance. (same as above)
    # sigma2. (same as above)
    # deltaE,gamma (same as above)
    # feff
    # path (possibly with quotes around it)
    #! blank line
    #! end of the single phase block
    #! after the specified number of phases, the phases block ends
    #! at the bottom there may be other stuff, namely the fitting procedure for minuit.
    #! this is ignored for now.


    with open(path, 'r') as f:
        _lines = f.readlines()
    lines = peekable(_lines)

    # Parse old file structure into dicts and lists
    _vars = _parse_variable_block(lines)
    _data = _parse_data_block(lines)
    _fitp = _parse_fitp_block(lines)
    _phss = _parse_all_phases_block(lines, numphases = _fitp['num_phases'])

    variables:dict[int, VariableSpec] = {
        varnum:VariableSpec(
            name=varname,
            initial=init,
            step=step,
            bounds=[lower, upper] if lower is not None and upper is not None else None,
            fix=(step == 0.0)
    ) for varnum, varname, init, step, lower, upper in _vars}

    phases = [PhaseSpec(
        name = f"phase{i+1}",
        position = Path(p['feff_path']),
        N = _resolve_var_reference(p['multiplicity'], variables),
        r = _resolve_var_reference(p['distance'], variables),
        s2 = _resolve_var_reference(p['sigma2'], variables),
        dE = _resolve_var_reference(p['deltae'], variables),
        s02 = None,
        eta = None,
    )
              for i,p in enumerate(_phss)]
    # Convert parsed data into new model structure
    model = ExafsModelSpecification(
        kind = 'feff',
        phasesroot = path.parent,
        variables = list(variables.values()),
        phases = phases,
        modelfile = path,
    )

    with open(Path(path).with_suffix('.estramodel'), 'w') as f:
        yaml.safe_dump(model.to_dict(), f, indent=4, encoding='utf-8', allow_unicode=True)

    return model
