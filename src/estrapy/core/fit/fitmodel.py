import yaml
import re
import numpy as np
import pandas as pd
from numpy import typing as npt

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Union, Self, Literal
from mashumaro import DataClassDictMixin
from collections import OrderedDict

from ...operations.fourier import flattop_window, fourier
from ...core.misc import Sack
from .feff8loader import FeffPath, load_path
from ...grammar.mathexpressions import Expression
from ...operations.axis_conversions import k_to_k1

VERSION_RE = re.compile(r'^#\s*version\s*:?\s*(?:(\d+)\.(\d+)(?:\.(\d+))?)\s*$', re.IGNORECASE)
VARIABLE_NAME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')

class ModelFormatError(Exception):
    """Exception raised for errors in the model format."""
    pass
class DependencyCycleError(Exception):
    """Exception raised for cycles in variable dependencies."""
    pass

@dataclass
class VariableSpec(DataClassDictMixin):
    name: str
    initial: Optional[float] = None
    bounds: Optional[List[float]] = None
    step: Optional[float] = None
    fix: Optional[bool] = None

@dataclass
class VariableRuleSpec(DataClassDictMixin):
    names: List[str]
    rule: str
    initial: Optional[List[float]] = None

@dataclass
class VariableCalcSpec(DataClassDictMixin):
    name: str
    expression: str

@dataclass
class PhaseSpec(DataClassDictMixin):
    name: str
    position: Optional[Path] = None
    N: Optional[str] = None
    rN: Optional[str] = None
    r: Optional[str] = None
    dr: Optional[str] = None
    dE: Optional[str] = None
    s2: Optional[str] = None
    s02: Optional[str] = None
    eta: Optional[str] = None

@dataclass
class ExafsModelSpecification(DataClassDictMixin):
    kind: str
    phasesroot: Path
    variables: List[Union[VariableSpec,VariableRuleSpec,VariableCalcSpec]] = field(default_factory=list[Union[VariableSpec,VariableRuleSpec,VariableCalcSpec]])
    phases: List[PhaseSpec] = field(default_factory=list[PhaseSpec])

    # To be populated when loading from file
    version: tuple[int, ...] = None # type: ignore
    modelfile: Path = None # type: ignore

def _load_model_spec(path: Path) -> ExafsModelSpecification:
    # Read the first line to check for # version
    # On missing version, attempt to read anyway, but warn the user
    try:
        model = _load_model_spec_from_yaml(path)
    except ModelFormatError:
        from ...legacy.model import load_model_from_legacy
        model = load_model_from_legacy(path)
    return model

def _load_model_spec_from_yaml(path: Path) -> ExafsModelSpecification:
    with open(path, "r") as f:
        first_line = f.readline()
        version_match = VERSION_RE.match(first_line)
        if version_match is None:
            if first_line.startswith('*'):
                raise ModelFormatError(f'Model specification file {path} appears to be in legacy format.')
            
            import warnings
            warnings.warn(f'Model specification file {path} is missing version header. Assuming correct version.')
            version = (0,0,0)
        else:
            version = tuple(int(x) if x is not None else 0 for x in version_match.groups())

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ModelFormatError(f'Error parsing YAML model specification file {path}: {e}') from e
    
    model = ExafsModelSpecification.from_dict(data) # type: ignore

    model.version = version
    model.modelfile = path

    return model

def save_model_spec_to_yaml(path: str, model: ExafsModelSpecification) -> None:
    yaml.safe_dump(model.to_dict(), open(path, 'w'), indent=4, encoding='utf-8', allow_unicode=True)

def rule_to_expression(varrule: VariableRuleSpec, rulename: str) -> tuple[dict[str,VariableSpec], dict[str,Expression[float]]]:
    """Converts parameters and rule into a set of base parametes with bounds and derived expressions.
    Ensure name is unique across the model, otherwise may overwrite existing variables."""
    
    base_vars:dict[str,VariableSpec] = {}
    expr_vars:dict[str,Expression[float]] = {}
    match varrule.rule:
        case 'simplex':
            # Adds n parameters (pₙ), defined from m = n-1 internal parameters
            # constraints: 0 <= pₙ <= 1
            #            : ∑ₙ pₙ = 1
            # internal parameters: norm = 1 + ∑ₘ exp(xₘ)
            #                      p₀ = 1 / norm
            #                      pₘ = exp(xₘ) / norm  for m = 1..(n-1)
            # The internal xₘ parameters have no bounds.
            nparams = len(varrule.names)
            if nparams < 2:
                raise ValueError(f'VariableRule {rulename} simplex rule requires at least two parameters.')
            
            if varrule.initial is not None:
                initialvals = varrule.initial
                if len(initialvals) != nparams:
                    raise ValueError(f'VariableRule {rulename} initial values length does not match names length.')
                if any(p < 0.0 for p in initialvals):
                    raise ValueError(f'VariableRule {rulename} initial values must be non-negative for simplex rule.')
                total = sum(initialvals)
                if total == 0.0:
                    raise ValueError(f'VariableRule {rulename} initial values cannot all be zero for simplex rule.')
                initialvals = [p / total for p in initialvals]
            else:
                initialvals = [1.0 / nparams for _ in varrule.names]

            # Add basevars
            for i in range(1, nparams):
                base_vars[f'_{rulename}_p{i}'] = VariableSpec(
                    name=f'_{rulename}_p{i}',
                    initial=np.log(initialvals[i] / initialvals[0]),
                    bounds=None,
                )

            # Add norm expression
            norm = f'_{rulename}_norm'
            expr_vars[f'_{rulename}_norm'] = Expression.compile('1 + ' + ' + '.join(f'exp(_{rulename}_p{i})' for i in range(1, nparams)))
            
            # Add expressions for each variable
            for i,param in enumerate(varrule.names):
                if i == 0:
                    expr = Expression[float].compile(f'1 / {norm}')
                else:
                    expr = Expression[float].compile(f'exp(_{rulename}_p{i}) / {norm}')
                expr_vars[param] = expr

            return base_vars, expr_vars
        
        # case 'vectornorm':
        #     # Adds n parameters (pₙ), defined from m = n-1 internal angles
        #     # constraints: ∑ₙ pₙ² = 1
        #     # internal parameters: angles θₘ
        #     #                      p₀ = cos(θ₀)
        #     #                      p₁ = sin(θ₀) * cos(θ₁)
        #     #                      p₂ = sin(θ₀) * sin(θ₁) * cos(θ₂)
        #     #                      ...
        #     nparams = len(varrule.names)
        #     if nparams < 2:
        #         raise ValueError(f'VariableRule {rulename} vectornorm rule requires at least two parameters.')
            
        #     if varrule.initial is not None:
        #         initialvals = varrule.initial
        #         if len(initialvals) != nparams:
        #             raise ValueError(f'VariableRule {rulename} initial values length does not match names length.')
        #         total_sq = sum(p*p for p in initialvals)
        #         if total_sq == 0.0:
        #             raise ValueError(f'VariableRule {rulename} initial values cannot all be zero for vectornorm rule.')
        #         initialvals = [p / np.sqrt(total_sq) for p in initialvals]
        #     else:
        #         initialvals = [1.0 / np.sqrt(nparams) for _ in varrule.names]
            
        #     # Add basevars
        #     for i in range(nparams - 1):
                

            
            
        case _:
            raise ValueError(f'Unknown variable rule: {varrule.rule}')

def _validate_model_spec_vars(model: ExafsModelSpecification) -> None:
    # Ensure all variable names are unique and valid
    varnames = set[str]()
    for variable in model.variables:
        match variable:
            case VariableSpec(name=name):
                if name in varnames:
                    raise ValueError(f'Duplicate variable name: {name}')
                # Check valid variable name
                if VARIABLE_NAME_RE.match(name) is None:
                    raise ValueError(f'Invalid variable name: {name}')
                varnames.add(name)
                # Also ensure that the variable initial value is within bounds if bounds are specified
                if variable.bounds is not None and variable.initial is not None:
                    if not (variable.bounds[0] <= variable.initial <= variable.bounds[1]):
                        raise ValueError(f'Variable {name} initial value {variable.initial} is outside of bounds {variable.bounds}.')
            case VariableRuleSpec(names=names):
                for name in names:
                    if name in varnames:
                        raise ValueError(f'Duplicate variable name: {name}')
                    # Check valid variable name
                    if VARIABLE_NAME_RE.match(name) is None:
                        raise ValueError(f'Invalid variable name in VariableRule: {name}')
                    varnames.add(name)
            case VariableCalcSpec(name=name):
                if name in varnames:
                    raise ValueError(f'Duplicate variable name: {name}')
                # Check valid variable name
                if VARIABLE_NAME_RE.match(name) is None:
                    raise ValueError(f'Invalid variable name: {name}')
                varnames.add(name)

    # Ensure all phase names are unique and valid
    phasenames = set[str]()
    for phase in model.phases:
        if phase.name in phasenames:
            raise ValueError(f'Duplicate phase name: {phase.name}')
        # Check valid phase name
        if VARIABLE_NAME_RE.match(phase.name) is None and phase.name != '<default>':
            raise ValueError(f'Invalid phase name: {phase.name}')
        phasenames.add(phase.name)


def _sort_expression_dependencies(exps: dict[str, Expression[float]], vars: dict[str, VariableSpec]) -> list[str]:
    # Build dependency graph and topologically sort expressions (we know vars are independent)
    graph: Sack[str, str] = Sack()
    for name, expr in exps.items():
        # Check required variables if the variable is defined
        # Raise error if any required variable is missing or if the variable requires itself
        missing = expr.required_vars - set(vars) - set(exps)
        if missing:
            raise ValueError(f'Undefined variable(s) in expression for variable \'{name}\': {", ".join(missing)}.')
        for reqvar in expr.required_vars:
            if reqvar == name:
                raise DependencyCycleError(f'Variable \'{name}\' depends on itself.')
            if reqvar in exps:
                graph.add(name, reqvar)
            else:
                graph.add_empty(name)
    
    # Topologically sort the graph
    sorted_exps: list[str] = []
    ready = sorted([n for n, deps in graph.groups() if not deps])
    graph.cull_empty()
    while ready:
        node = ready.pop(0)
        sorted_exps.append(node)

        for neighbor,neighbordeps in [*graph.groups()]:
            if node in neighbordeps:
                graph.remove(neighbor, node)
                if neighbor not in graph:
                    ready.append(neighbor)
                    ready.sort()
    
    if graph:
        formatted = "\n".join(f'  {n} <- {", ".join(deps)}' for n,deps in graph.groups())
        raise DependencyCycleError(f'Cycle detected in variable dependencies:\n{formatted}')
    
    return sorted_exps

@dataclass
class Phase:
    feffpath: FeffPath
    N: Expression[float]
    r: Expression[float]
    dE: Expression[float]
    s02: Expression[float]
    eta: Expression[float]
    # Gaussian Debye-Waller factor
    s2: Expression[float]
    # TODO: implement non-Gaussian DW factors
    # Asymmetry parameter for non-Gaussian DW factor
    # beta: Expression
    # # Cumulant expansion
    # p3: Expression
    # p4: Expression

    kind: Literal['gaussian', 'cumulant', 'asymmetric'] = 'gaussian'

    def chi(self, k: npt.NDArray[np.float64], kpow: float = 0, **kw: float) -> npt.NDArray[np.float64]:
        match self.kind:
            case 'gaussian':
                return self._chi_gaussian(k, kpow, **kw)
            # case 'cumulant':
            #     return self._chi_cumulant(k, kpow, **kw)
            # case 'asymmetric':
            #     return self._chi_asymmetric(k, kpow, **kw)
            case _:
                raise ValueError(f'Unknown phase kind: {self.kind}')

    def _chi_gaussian(self, k: npt.NDArray[np.float64], kpow: float = 0, **kw: float) -> npt.NDArray[np.float64]:
        """Compute the EXAFS contribution for this phase at given k values."""
        # Calculate shifted k values according to dE
        k1 = k_to_k1(k, self.dE(**kw))

        s02 = self.s02(**kw)
        N = self.N(**kw)
        r = self.r(**kw)
        s2 = self.s2(**kw)
        eta = self.eta(**kw)

        amp: npt.NDArray[np.floating] = np.asarray(self.feffpath.interp['amp'](k1)) # type: ignore
        phs: npt.NDArray[np.floating] = np.asarray(self.feffpath.interp['phs'](k1)) # type: ignore
        lam: npt.NDArray[np.floating] = np.asarray(self.feffpath.interp['lambda'](k1)) # type: ignore

        # If kpow = 0, divide by k. If kpow = 1, do not divide. If kpow = 2, multiply by k^1.
        pre = s02 * N / (r * r)
        kfc = np.ones_like(k1) if kpow == 1 else k**(kpow - 1)

        osc = np.sin(2 * k * r + phs + 2*eta)
        fre = np.exp(-2 * r / lam)
        dec = np.exp(-2 * k * k * s2)

        chi = pre * kfc * amp * fre * dec * osc
        return chi
    
    
    def required_vars(self) -> set[str]:
        reqvars = set[str]()
        reqvars.update(self.N.required_vars)
        reqvars.update(self.r.required_vars)
        reqvars.update(self.dE.required_vars)
        reqvars.update(self.s2.required_vars)
        reqvars.update(self.s02.required_vars)
        reqvars.update(self.eta.required_vars)
        return reqvars
    
    @classmethod
    def from_spec(cls, spec: PhaseSpec, available_vars: set[str], defaultphase: PhaseSpec, *, rootdir: Path) -> Self:
        if spec.position is None:
            raise ValueError(f'Phase {spec.name} is missing position to Feff path file.')
        
        if spec.position.is_absolute(): file = spec.position.resolve()
        else: file = (rootdir / spec.position).resolve()
        if not file.exists():
            raise ValueError(f'Phase file does not exist: {file}')
        
        # Load the FeffPath
        feffpath = load_path(file)

        # Prepare parameter expressions
        match spec.N, spec.rN, defaultphase.N, defaultphase.rN:
            case None, None, None, None:
                N_expr = Expression[float].compile('1.0')

            case str(Nstr), None, _, _:
                N_expr = Expression[float].compile(Nstr)
            case None, str(rNstr), _, _:
                N_expr = Expression[float].compile(f'({rNstr}) * {feffpath.mult}')

            case None, None, dNstr, None:
                N_expr = Expression[float].compile(dNstr)
            case None, None, None, drNstr:
                N_expr = Expression[float].compile(f'({drNstr}) * {feffpath.mult}')

            case _:
                raise ValueError(f'Phase {spec.name} is ill defined (invalid multiplicity).')
            
        if N_expr.required_vars - available_vars:
            missing = N_expr.required_vars - available_vars
            raise ValueError(f'Undefined variable(s) in N expression for phase {spec.name}: {", ".join(missing)}.')

        match spec.r, spec.dr, defaultphase.r, defaultphase.dr:
            case None, None, None, None:
                r_expr = Expression[float].compile(str(feffpath.reff))
            
            case str(rstr), None, _, _:
                r_expr = Expression[float].compile(rstr)
            case None, str(drstr), _, _:
                r_expr = Expression[float].compile(f'({drstr}) + {feffpath.reff}')
            
            case None, None, str(drstr), None:
                r_expr = Expression[float].compile(f'({drstr}) + {feffpath.reff}')
            case None, None, None, str(rstr):
                r_expr = Expression[float].compile(rstr)
            
            case _:
                raise ValueError(f'Phase {spec.name} is ill defined (invalid distance).')
            
        if r_expr.required_vars - available_vars:
            missing = r_expr.required_vars - available_vars
            raise ValueError(f'Undefined variable(s) in r expression for phase {spec.name}: {", ".join(missing)}.')
        
        match spec.dE, defaultphase.dE:
            case str(dEstr), _:
                dE_expr = Expression[float].compile(dEstr)
            case None, str(dEstr):
                dE_expr = Expression[float].compile(dEstr)
            case None, None:
                dE_expr = Expression[float].compile('0.0')
        
        if dE_expr.required_vars - available_vars:
            missing = dE_expr.required_vars - available_vars
            raise ValueError(f'Undefined variable(s) in dE expression for phase {spec.name}: {", ".join(missing)}.')
        
        match spec.s2, defaultphase.s2:
            case str(s2str), _:
                s2_expr = Expression[float].compile(s2str)
            case None, str(s2str):
                s2_expr = Expression[float].compile(s2str)
            case None, None:
                s2_expr = Expression[float].compile('1.0')
        
        if s2_expr.required_vars - available_vars:
            missing = s2_expr.required_vars - available_vars
            raise ValueError(f'Undefined variable(s) in s2 expression for phase {spec.name}: {", ".join(missing)}.')
        
        match spec.s02, defaultphase.s02:
            case str(s02str), _:
                s02_expr = Expression[float].compile(s02str)
            case None, str(s02str):
                s02_expr = Expression[float].compile(s02str)
            case None, None:
                s02_expr = Expression[float].compile('1.0')
        
        if s02_expr.required_vars - available_vars:
            missing = s02_expr.required_vars - available_vars
            raise ValueError(f'Undefined variable(s) in s02 expression for phase {spec.name}: {", ".join(missing)}.')
        
        match spec.eta, defaultphase.eta:
            case str(etastr), _:
                eta_expr = Expression[float].compile(etastr)
            case None, str(etastr):
                eta_expr = Expression[float].compile(etastr)
            case None, None:
                eta_expr = Expression[float].compile('0.0')
        
        if eta_expr.required_vars - available_vars:
            missing = eta_expr.required_vars - available_vars
            raise ValueError(f'Undefined variable(s) in eta expression for phase {spec.name}: {", ".join(missing)}.')
        
        return cls(
            feffpath=feffpath,
            N=N_expr,
            r=r_expr,
            dE=dE_expr,
            s2=s2_expr,
            s02=s02_expr,
            eta=eta_expr
        )
        
    
@dataclass(slots=True)
class ExafsFitResult:
    fitted_vars: dict[str, float]
    chi_squared: float
    reduced_chi_squared: float
    fittedchi: npt.NDArray[np.float64]
    residuals: npt.NDArray[np.float64]
    partialphases: dict[str, npt.NDArray[np.float64]]

from matplotlib import pyplot as plt

@dataclass
class ExafsModel:
    phases: List[Phase]
    vars: dict[str, VariableSpec]
    exps: OrderedDict[str, Expression[float]]

    @classmethod
    def load(cls, path:Path) -> Self:
        spec = _load_model_spec(path)
        _validate_model_spec_vars(spec)

        vars: dict[str, VariableSpec] = {}
        exps: dict[str, Expression[float]] = {}
        for variable in spec.variables:
            match variable:
                case VariableSpec():
                    vars[variable.name] = variable
                case VariableRuleSpec():
                    base_vars, expr_vars = rule_to_expression(variable, f'rule_{''.join(variable.names)}')
                    vars.update(base_vars)
                    exps.update(expr_vars)
                case VariableCalcSpec():
                    expression = Expression[float].compile(variable.expression)
                    exps[variable.name] = expression
        
        available_varnames = set(vars) | set(exps)
        sorted_exps = _sort_expression_dependencies(exps, vars)
        used_varnames = set[str]().union(*[e.required_vars for e in exps.values()])

        phases: list[Phase] = []
        # Get phase with name <default> if any
        default_phase_spec = next((p for p in spec.phases if p.name == '<default>'),
                                  PhaseSpec(name='<default>'))
        
        for phasespec in spec.phases:
            if phasespec.name == '<default>':
                continue
            phase = Phase.from_spec(phasespec, available_varnames, defaultphase=default_phase_spec, rootdir=spec.phasesroot)
            used_varnames.update(phase.required_vars())
            
            phases.append(phase)

        if (unused := available_varnames - used_varnames):
            raise ValueError(f'Unused variable(s) in model specification: {", ".join(unused)}.')
        
        return cls(
            phases=phases,
            vars=vars,
            exps=OrderedDict((name, exps[name]) for name in sorted_exps)
        )

    def calc(self, k: npt.NDArray[np.float64], vars: dict[str, float], kweight: float = 0.0) -> npt.NDArray[np.float64]:
        """Compute the total EXAFS signal for the model at given k values and with the given variable values."""
        chi_total = np.zeros_like(k, dtype=np.float64)
        for phase in self.phases:
            chi_total += phase.chi(k, kweight, **vars)
        return chi_total
    
    def _get_initials(self, *, default: float = 0.0) -> dict[str, float]:
        """Get a dictionary of variable initial values."""
        return {varname:var.initial if var.initial is not None else default for varname,var in self.vars.items()}

    def _get_exps(self, vars: dict[str, float], *, inplace:bool=False) -> dict[str, float]:
        """Return a new dictionary with expression variables evaluated and added, without modifying vars."""
        newvars = vars if inplace else vars.copy()
        for expname,exp in self.exps.items():
            newvars[expname] = exp(**newvars)
        return newvars
    
    def fit_k(self, k: npt.NDArray[np.float64], chi: npt.NDArray[np.float64], krange: tuple[float, float], kweight: float) -> ExafsFitResult:
        initial = self._get_initials()
        bounds = (
            [self.vars[varname].bounds[0] if self.vars[varname].bounds is not None else -np.inf for varname in initial.keys()],
            [self.vars[varname].bounds[1] if self.vars[varname].bounds is not None else np.inf for varname in initial.keys()],
        )
        vars = self._get_exps(initial)

        kidx = np.where((k >= krange[0]) & (k <= krange[1]))[0]
        kfit = k[kidx]
        chifit = chi[kidx] * (kfit ** kweight)

        # weight vector given by the spacing in x
        weight = np.sqrt(np.gradient(kfit))

        from scipy.optimize import least_squares, OptimizeResult, minimize
        
        history: list[OptimizeResult] = []
        def cb(intermediateresult: OptimizeResult) -> None:
            history.append(intermediateresult)
        
        def residuals(varvalues: npt.NDArray[np.float64]):
            vars = dict(zip(initial.keys(), varvalues))
            self._get_exps(vars, inplace=True)
            chicalc = self.calc(kfit, vars, kweight)
            residual = chicalc - chifit
            weighted = residual * weight

            return weighted

        result: OptimizeResult = least_squares(
            fun=residuals,
            x0=np.array([initial[varname] for varname in self.vars.keys()]),
            bounds=bounds,
            jac='3-point',
            callback=cb,
        )

        vars:dict[str, float] = dict(zip(self.vars.keys(), result.x))
        self._get_exps(vars, inplace=True)

        _calc = self.calc(kfit, vars, kweight)
        vshift = np.ptp(_calc)/2
        plt.scatter(kfit, chifit, color="green", marker="+", linewidth=0.5, s=20)
        plt.plot(kfit, _calc, color="black")

        for idx,phas in enumerate(self.phases, start=1):
            part = phas.chi(kfit, kweight, **vars)
            plt.plot(kfit, part - idx * vshift, label=f'Phase {idx}', color="orange")
        
        plt.scatter(kfit, chifit - _calc - (idx+2) * vshift, color="black", linewidth=0.5, s=20, marker= "o", facecolors='none')

        window = flattop_window(kfit, (kfit[0], kfit[0]+0.5, kfit[-1]-0.5, kfit[-1]), 'hanning')

        r = np.linspace(0, 7, 1000)
        fdat = fourier(kfit, chifit * window, r)
        fchi = fourier(kfit, _calc * window, r)

        vshift = np.ptp(fchi.real)/2
        plt.figure()
        plt.scatter(r, np.abs(fdat), color="green", marker="+", linewidth=0.5, s=20, label='Data')
        plt.scatter(r, np.imag(fdat), color="green", marker="x", linewidth=0.5, s=20)

        plt.plot(r, np.abs(fchi), color="black", label='Fit')
        plt.plot(r, np.imag(fchi), color="black")

        for idx,phas in enumerate(self.phases, start=1):
            part = fourier(kfit, phas.chi(kfit, kweight, **vars) * window, r)
            plt.plot(r, np.imag(part) - idx * vshift, label=f'Phase {idx}', color="orange")
            plt.plot(r, np.abs(part) - idx * vshift, label=f'Phase {idx}', color="orange", linewidth=0.5)
            plt.plot(r, -np.abs(part) - idx * vshift, label=f'Phase {idx}', color="orange", linewidth=0.5)
        
        plt.scatter(r, np.imag(fdat - fchi) - (idx+2) * vshift, color="black", linewidth=0.5, s=20, marker= "o", facecolors='none')

        return ExafsFitResult(
            fitted_vars=vars,
            chi_squared=float(result.cost * 2),
            reduced_chi_squared=float(result.cost * 2 / (len(kfit) - len(self.vars))),
            fittedchi=self.calc(kfit, vars, kweight),
            residuals=residuals(result.x),
            partialphases={f'phase_{i}': phase.chi(kfit, kweight, **vars) for i,phase in enumerate(self.phases)},
        )