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
from scipy.optimize import least_squares, OptimizeResult #, approx_fprime # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from scipy.optimize._numdiff import approx_derivative # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from .feff8loader import FeffPath, load_path
from ..misc import Sack
from ..grammar.mathexpressions import Expression
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
        from .legacy import load_model_from_legacy
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
        if expr.required_vars:
            for reqvar in expr.required_vars:
                if reqvar == name:
                    raise DependencyCycleError(f'Variable \'{name}\' depends on itself.')
                if reqvar in exps:
                    graph.add(name, reqvar)
                else:
                    graph.add_empty(name)
        else:
            # The variable has no dependencies and is a fixed value
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
    name: str
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
            name = spec.name,
            feffpath=feffpath,
            N=N_expr,
            r=r_expr,
            dE=dE_expr,
            s2=s2_expr,
            s02=s02_expr,
            eta=eta_expr,
        )
        
    
@dataclass(slots=True)
class ExafsFitResult:
    # Resulting parameters of the fit
    fitted_vars: pd.Series
    kaxis: npt.NDArray[np.floating]
    fittedknchi: npt.NDArray[np.floating]
    partialknphases: dict[str, npt.NDArray[np.floating]]

    # Internal storage of fit details
    _param_order: list[str]
    _fitted_params: npt.NDArray[np.floating]
    _expressions: OrderedDict[str, Expression[float]]

    # Covariance and correlation matrices
    std_errors: pd.Series
    covariance: pd.DataFrame
    correlation: pd.DataFrame

    # Fit parameters
    weights: npt.NDArray[np.floating]
    krange: tuple[float, float]

    # Fit statistics
    residuals: npt.NDArray[np.floating]
    difference: npt.NDArray[np.floating]
    chi_squared: float
    reduced_chi_squared: float
    R_squared: float
    reduced_R_squared: float
    dof: float
    n_params: int
    n_points: int
    residual_variance: float

    # Optimization details
    finalresult: OptimizeResult
    history: list[OptimizeResult]

    # Original data
    kdata: npt.NDArray[np.floating]
    knchidata: npt.NDArray[np.floating]
    kpow: float


def jacobian_finite_difference(
        param_names: list[str],
        params: npt.NDArray[np.floating],
        expressions: OrderedDict[str, Expression[float]]
    ) -> npt.NDArray[np.floating]:
    """
    Compute Jacobian of expressions with respect to parameters using finite differences.
    
    Parameters
    ----------
    params : np.ndarray
        Independent parameter values (ordered)
    param_names : list[str]
        Names of parameters (same order as params)
    expressions : OrderedDict[str, Expression[float]]
        Dependent parameter expressions (ordered by dependency)
    
    Returns
    -------
    J : np.ndarray
        Jacobian matrix (n_expressions × n_params)
    """
    # Create a vectorized function that evaluates all expressions
    def vector_func(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        # Reconstruct params dict from array
        vars_dict = dict(zip(param_names, x))
        for n,expr in expressions.items():
            vars_dict[n] = expr(**vars_dict)
        return np.array([vars_dict[n] for n in expressions.keys()])
    
    xk = np.asarray(params, float)
    f0 = vector_func(xk)

    # Using approx_derivative from scipy for better accuracy with complex step method
    return np.array(approx_derivative(vector_func, xk, method='cs', f0=f0))
    return np.array(approx_fprime(params, vector_func), dtype=np.float64)

def sanitize_covariance_matrix(
    cov_matrix: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Sanitize covariance matrix for numerical stability and symmetry.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix to sanitize
    
    Returns
    -------
    cov_sanitized : np.ndarray
        Sanitized covariance matrix
    """

    # Bool mask of values that are to be set to zero, for numerical stability and symmetry
    iszero = np.isclose(cov_matrix, 0.0, atol=1e-15)
    
    # Check for negative variances on the diagonal and set them to zero, along with their rows and columns
    diagzero = np.diag(cov_matrix) < 0
    iszero[diagzero, :] = True
    iszero[:, diagzero] = True

    # Ensure symmetry of the iszero mask
    iszero |= iszero.T

    # Create a copy to avoid modifying the original
    cov_sanitized = np.array(cov_matrix, copy=True)
    
    # Set identified values to zero
    cov_sanitized[iszero] = 0.0

    # Force symmetry
    np.add(cov_sanitized, cov_sanitized.T, out=cov_sanitized)
    cov_sanitized *= 0.5

    return cov_sanitized

def extend_covariance_matrix(
    cov_matrix: npt.NDArray[np.floating],
    param_names: list[str],
    params: npt.NDArray[np.floating],
    expressions: OrderedDict[str, Expression[float]]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], list[str]]:
    """
    Extend covariance and correlation matrices to include dependent parameters.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix of independent parameters
    param_names : list[str]
        Names of independent parameters (ordered)
    params : np.ndarray
        Independent parameter values (ordered)
    expressions : OrderedDict[str, Expression[float]]
        Dependent parameter expressions (ordered by dependency)
    
    Returns
    -------
    cov_combined : np.ndarray
        Combined covariance matrix
    corr_combined : np.ndarray
        Combined correlation matrix
    all_names : list[str]
        Combined parameter names (independent + dependent)
    """
    # Compute Jacobian using scipy
    J_dep = jacobian_finite_difference(param_names, params, expressions)
    expr_names = list(expressions.keys())
    
    # Propagate covariance to dependent parameters
    # Cov(y) = J @ Cov(x) @ J^T
    cov_dep = J_dep @ cov_matrix @ J_dep.T
    
    # Create combined covariance matrix
    all_names = param_names + expr_names
    n_indep = len(param_names)
    n_dep = len(expr_names)
    n_total = n_indep + n_dep
    
    cov_combined = np.zeros((n_total, n_total))
    
    # Independent-independent block (top-left)
    cov_combined[:n_indep, :n_indep] = cov_matrix
    
    # Dependent-dependent block (bottom-right)
    cov_combined[n_indep:, n_indep:] = cov_dep
    
    # Cross-covariance blocks
    # Cov(x, y) = Cov(x) @ J^T
    cov_cross = cov_matrix @ J_dep.T
    cov_combined[:n_indep, n_indep:] = cov_cross
    cov_combined[n_indep:, :n_indep] = cov_cross.T

    cov_combined = sanitize_covariance_matrix(cov_combined)

    std_dev_scaling = np.sqrt(np.diag(cov_combined))
    
    # Handle zero standard deviations (setting to 1 is ok because their correlation is zero)
    std_dev_scaling = np.where(std_dev_scaling > 0, std_dev_scaling, 1.0)
    
    D_inv_combined = np.diag(1.0 / std_dev_scaling)
    corr_combined = D_inv_combined @ cov_combined @ D_inv_combined
    
    return cov_combined, corr_combined, all_names

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
                    if variable.fix:
                        fixvar = Expression[float].compile(str(variable.initial if variable.initial is not None else 0.0))
                        exps[variable.name] = fixvar
                    else:
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
    
    def _get_initials(self, order: list[str], *, default: float = 0.0) -> npt.NDArray[np.floating]:
        """Get a dictionary of variable initial values."""
        return np.array([self.vars[var].initial if self.vars[var].initial is not None else default for var in order], dtype=np.float64)
    
    def _get_bounds(self, order: list[str]) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
        """Get lower and upper bounds for variables as two arrays."""
        bounds = np.array([self.vars[var].bounds if self.vars[var].bounds is not None else [-np.inf, np.inf] for var in order], dtype=np.float64)
        return bounds[:,0], bounds[:,1]

    def _get_exps(self, vars: dict[str, float], *, inplace:bool=False) -> dict[str, float]:
        """Return a new dictionary with expression variables evaluated and added, without modifying vars."""
        newvars = vars if inplace else vars.copy()
        for expname,exp in self.exps.items():
            newvars[expname] = exp(**newvars)
        return newvars
    
    def fit_k(self, k: npt.NDArray[np.float64], chi: npt.NDArray[np.float64], krange: tuple[float, float], kweight: float) -> ExafsFitResult:
        param_order = list(self.vars.keys())
        initial, bounds = self._get_initials(param_order), self._get_bounds(param_order)

        kidx = np.where((k >= krange[0]) & (k <= krange[1]))[0]
        kfit = k[kidx]
        chifit = chi[kidx] * (kfit ** kweight)

        # weight vector given by the spacing in x
        dk_weight = np.sqrt(np.gradient(kfit))
        
        history: list[OptimizeResult] = []
        def cb(intermediate_result: OptimizeResult) -> None:
            history.append(intermediate_result)
        
        def residuals(varvalues: npt.NDArray[np.float64]):
            vars = dict(zip(param_order, varvalues))
            self._get_exps(vars, inplace=True)
            chicalc = self.calc(kfit, vars, kweight)
            residual = chicalc - chifit
            weighted = residual * dk_weight

            return weighted

        result: OptimizeResult = least_squares( # pyright: ignore[reportUnknownVariableType]
            fun=residuals,
            x0=initial,
            bounds=bounds,
            jac='3-point',
            callback=cb,
        )
        assert isinstance(result, OptimizeResult)

        # Covariance and correlation matrices --------------------------------------------------------------
        JTJ:npt.NDArray[np.floating] = np.asarray(result.jac.T @ result.jac) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        residual:npt.NDArray[np.floating] = np.asarray(result.fun) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        parameters:npt.NDArray[np.floating] = np.asarray(result.x) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

        # Estimate the residual variance
        # degrees of freedom = number of observations - number of parameters
        n_obs, n_params = len(residual), len(parameters)
        dof = n_obs - n_params
        residual_variance = np.sum(residual**2) / dof

        # Covariance matrix (inverse of J^T * J, scaled by residual variance)
        try:
            cov_matrix = np.linalg.inv(JTJ) * residual_variance
        except np.linalg.LinAlgError:
            # If singular, use pseudoinverse
            cov_matrix = np.linalg.pinv(JTJ) * residual_variance

        cov_matrix, corr_matrix, value_order = extend_covariance_matrix(cov_matrix, param_order, parameters, self.exps)
        _fitted_param_dict: dict[str, float] = dict(zip(param_order, parameters))
        self._get_exps(_fitted_param_dict, inplace=True)
        fitted_params = np.array([_fitted_param_dict[name] for name in value_order], dtype=np.float64)
        _cov_df = pd.DataFrame(cov_matrix, index=value_order, columns=value_order)
        _corr_df = pd.DataFrame(corr_matrix, index=value_order, columns=value_order)
        
        # TODO: come fa la varianza ad essere negativa lo sa solo il diavolo, però succede.
        # Ma che cazzo, numpy?
        _std_err = pd.Series(np.sqrt(np.diag(cov_matrix)), index=value_order)

        external_pars = [v for v in value_order if not v.startswith("_")]

        # Prepare fit result --------------------------------------------------------------------------------
        partial_phases: dict[str, npt.NDArray[np.floating]] = {
            phase.name: phase.chi(kfit, kweight, **_fitted_param_dict) for phase in self.phases
        }
        fittedknchi = self.calc(kfit, _fitted_param_dict, kweight)


        params = pd.Series(
            data=[_fitted_param_dict[name] for name in external_pars],
            index=external_pars
        )

        cov_df = _cov_df.loc[external_pars, external_pars]
        corr_df = _corr_df.loc[external_pars, external_pars]
        std_err = _std_err.loc[external_pars]

        # Regression statistics -----------------------------------------------------------------------
        chi2 = float(np.sum(residual**2))
        tss = np.sum((chifit - np.mean(chifit))**2)
        r2 = float(1.0 - chi2 / tss)
        red_chi2 = float(chi2 / dof)
        red_r2 = float(1.0 - (chi2 / dof) / (tss / (n_obs - 1)))

        return ExafsFitResult(
            fitted_vars = params,
            kaxis = kfit,
            fittedknchi = fittedknchi,
            partialknphases = partial_phases,

            _param_order = param_order,
            _fitted_params = fitted_params,
            _expressions = self.exps,

            std_errors = std_err,
            covariance = cov_df,
            correlation = corr_df,

            weights = dk_weight,
            krange = krange,

            residuals = residual,
            difference = chifit - fittedknchi,
            chi_squared = chi2,
            reduced_chi_squared = red_chi2,
            R_squared = r2,
            reduced_R_squared = red_r2,
            dof = dof,
            n_params = n_params,
            n_points = n_obs,
            residual_variance = float(residual_variance),
            
            finalresult = result,
            history = history,
            
            kdata = kfit,
            knchidata = chifit,
            kpow = kweight
        )
