"""Configuration and result dataclasses for IRMTR."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Config:
    alpha: float=1e-4
    beta: float=1e-2
    desired_state: float=-1.
    eta1: float=1e-4
    eta2: float=.75
    gamma_shrink: float=.5
    gamma_expand: float=2.
    kappa_fcd: float=.1
    curvature: float=.1  # Positive baseline Hessian curvature (L2 geometry)
    delta_local: float=100.
    delta_max: float=100.
    tau_delta: float=.05
    e_fraction: float=.1 # prescribed e_k^ell = e_fraction*tau_j
    state_tol: float=1e-9
    adjoint_tol: float=1e-9
    max_inner: int=400
    max_coarse_inner: int=1
    kappa_ml: float=.5
    kappa_et: float=.1
    max_outer: int=10
    verbose: bool=True
    smooth_only: bool = False
    hessian_mode: str = 'diagonal'  # 'diagonal', 'lbfgs', or 'exact'
    lbfgs_memory: int = 5
    lbfgs_cg_maxiter: int = 50
    lbfgs_cg_rtol: float = 1e-3
    progress_every: int = 25


@dataclass
class Evaluation:
    F: float
    g: np.ndarray  # inexact gradient of f
    G: np.ndarray  # inexact gradient of f + exact Moreau gradient
    e: float       # prescribed error budget (not numerically certified)
    u: np.ndarray = None  # cached state for the exact Hessian
    p: np.ndarray = None  # cached adjoint for the exact Hessian
    J: object = None      # same nonlinear Jacobian as converged state


@dataclass
class Level:
    mesh: object
    R: object=None  # Restriction from this level to previous level
    P: object=None  # L2-adjoint prolongation from previous to this level


@dataclass
class InnerResult:
    z: np.ndarray
    iterations: int
    residual_bound: float
    converged: bool
    reason: str
    # Valid for a single-step coarse call using its parent-provided Bc.
    # It is the reduction of the CHILD'S OWN quadratic model, not the
    # lower-level predicted reduction used in the acceptance ratio.
    local_predicted: float = None


@dataclass
class OuterResult:
    z: np.ndarray
    history: list=field(default_factory=list)
    converged: bool=False


