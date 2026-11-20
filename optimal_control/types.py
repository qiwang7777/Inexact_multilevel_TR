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
    e_fraction: float=.1 # deprecated legacy field; no longer used for certification
    kappa_g: float=.10 # certified dynamic gradient forcing coefficient
    state_tol: float=1e-9
    adjoint_tol: float=1e-9
    max_inner: int=400
    max_coarse_inner: int=1
    kappa_ml: float=.5
    kappa_et: float=.1
    max_outer: int=10
    verbose: bool=True
    smooth_only: bool = False
    hessian_mode: str = 'exact'  # 'exact' is default; 'lbfgs', 'diagonal' optional
    lbfgs_memory: int = 5
    lbfgs_cg_maxiter: int = 50
    lbfgs_cg_rtol: float = 1e-3
    progress_every: int = 25
    fine_every: int = 0  # optional forced finest-level fine step (0 disables)


@dataclass
class Evaluation:
    F: float
    g: np.ndarray  # inexact gradient of f
    G: np.ndarray  # inexact gradient of f + exact Moreau gradient
    e: float       # PDE-residual-certified bound on discrete reduced-gradient error
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

    @property
    def residual(self):
        return self.residual_bound


@dataclass
class OuterResult:
    z: np.ndarray
    history: list=field(default_factory=list)
    converged: bool=False



    @property
    def stages(self):
        """Alias: elliptic reporting expects stages; semilinear expects history."""
        return self.history
