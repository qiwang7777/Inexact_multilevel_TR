# Full semilinear PDE control with Moreau-Yosida regularization.
# Default: exact reduced PDE Hessian plus generalized MY Hessian,
#         full outer continuation and original-problem stationarity test.
# Optional modes: diagonal, L-BFGS, fixed-mu diagnostic, smooth-only diagnostic.
from dataclasses import dataclass, field
import argparse
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class TriMesh:
    nodes: np.ndarray
    elements: np.ndarray
    boundary: np.ndarray
    interior: np.ndarray


def create_unit_square_mesh(n):
    """P1 triangle nodes; P0 control: 2*n*n elements."""
    x = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    nodes = np.column_stack((X.ravel(), Y.ravel()))
    idx = lambda i, j: i * (n + 1) + j
    elements = np.array(
        [tri for i in range(n) for j in range(n)
         for tri in ([idx(i,j), idx(i+1,j), idx(i+1,j+1)],
                     [idx(i,j), idx(i+1,j+1), idx(i,j+1)])],
        dtype=int,
    )
    boundary = ((nodes[:, 0] == 0) | (nodes[:, 0] == 1)
                | (nodes[:, 1] == 0) | (nodes[:, 1] == 1))
    return TriMesh(nodes, elements, boundary, np.flatnonzero(~boundary))


def element_geometry(mesh):
    if hasattr(mesh,'_geometry_cache'):
        return mesh._geometry_cache
    pts = mesh.nodes[mesh.elements]
    a, b, c = pts[:, 0], pts[:, 1], pts[:, 2]
    det = ((b[:,0]-a[:,0])*(c[:,1]-a[:,1])
           -(c[:,0]-a[:,0])*(b[:,1]-a[:,1]))
    grads = np.stack((
        np.column_stack((b[:,1]-c[:,1], c[:,0]-b[:,0])),
        np.column_stack((c[:,1]-a[:,1], a[:,0]-c[:,0])),
        np.column_stack((a[:,1]-b[:,1], b[:,0]-a[:,0])),
    ), axis=1) / det[:,None,None]
    mesh._geometry_cache = (np.abs(det)/2.0,grads)
    return mesh._geometry_cache


def triangle_quadrature():
    bary = np.array([
        [1/3, 1/3, 1/3],
        [.059715871789770, .470142064105115, .470142064105115],
        [.470142064105115, .059715871789770, .470142064105115],
        [.470142064105115, .470142064105115, .059715871789770],
        [.797426985353087, .101286507323456, .101286507323456],
        [.101286507323456, .797426985353087, .101286507323456],
        [.101286507323456, .101286507323456, .797426985353087],
    ])
    weights = np.array([.225] + [.132394152788506]*3
                             + [.125939180544827]*3)
    return bary, weights


def assemble_stiffness(mesh):
    areas, grads = element_geometry(mesh)
    E = mesh.elements
    Kloc = areas[:,None,None] * np.einsum('eik,ejk->eij',grads,grads)
    return sp.coo_matrix((Kloc.ravel(),
                          (np.repeat(E,3,axis=1).ravel(),
                           np.tile(E,(1,3)).ravel())),
                         shape=(len(mesh.nodes),)*2).tocsr()

def get_stiffness(mesh):
    if not hasattr(mesh, '_K_cached'):
        mesh._K_cached = assemble_stiffness(mesh)
    return mesh._K_cached 


def assemble_control_load(mesh,z):
    areas, _ = element_geometry(mesh)
    if len(z)!=len(areas):
        raise ValueError('P0 control must have one coefficient per triangle')
    return np.bincount(mesh.elements.ravel(),
                       weights=np.repeat(areas*z/3.0,3),
                       minlength=len(mesh.nodes))


def assemble_nonlinear_terms(mesh,u):
    areas, _ = element_geometry(mesh)
    bary, weights = triangle_quadrature()
    E = mesh.elements
    uq = u[E] @ bary.T
    Nloc = np.einsum('eq,q,qi,e->ei', uq**3, weights, bary, areas)
    N = np.bincount(E.ravel(), weights=Nloc.ravel(), minlength=len(mesh.nodes))
    Jloc = np.einsum('eq,q,qi,qj,e->eij',3.0*uq**2,weights,bary,bary,areas)
    J = sp.coo_matrix((Jloc.ravel(),
                       (np.repeat(E,3,axis=1).ravel(),
                        np.tile(E,(1,3)).ravel())),
                      shape=(len(mesh.nodes),)*2).tocsr()
    return N,J


def solve_state(mesh,z,tol=1e-8,max_newton=30,verbose=False,u0=None,
                return_jacobian=False):
    #K=assemble_stiffness(mesh)
    K = get_stiffness(mesh)
    rhs=assemble_control_load(mesh,z)
    I=mesh.interior
    u=np.zeros(len(mesh.nodes)) if u0 is None else np.array(u0,dtype=float,copy=True)
    u[mesh.boundary]=0.0
    for it in range(max_newton):
        nonlinear,J=assemble_nonlinear_terms(mesh,u)
        r=(K@u+nonlinear-rhs)[I]
        nr=np.linalg.norm(r)
        if verbose: print(f'State Newton {it}: residual {nr:.3e}')
        if nr<=tol: return (u, J) if return_jacobian else u
        du=spla.spsolve((K+J)[I][:,I],-r)
        u[I]+=du
        u[mesh.boundary]=0.0
    raise RuntimeError('State Newton iteration did not converge')


def assemble_p1_load(mesh,u,w_value=-1.0):
    areas,_=element_geometry(mesh)
    bary,weights=triangle_quadrature()
    uq=u[mesh.elements] @ bary.T
    bloc=np.einsum('eq,q,qi,e->ei',w_value-uq,weights,bary,areas)
    return np.bincount(mesh.elements.ravel(),weights=bloc.ravel(),
                       minlength=len(mesh.nodes))


def solve_adjoint(mesh,u,w_value=-1.0,rtol=1e-8,maxiter=1000,verbose=False,
                  J=None,p0=None):
    #K=assemble_stiffness(mesh)
    K = get_stiffness(mesh)
    if J is None:
        _,J=assemble_nonlinear_terms(mesh,u)
    I=mesh.interior
    A=(K+J)[I][:,I]
    b=assemble_p1_load(mesh,u,w_value)[I]
    x0 = None if p0 is None else np.asarray(p0)[I]
    pI,info=spla.cg(A,b,x0=x0,rtol=rtol,atol=0.0,maxiter=maxiter)
    if info<0: raise RuntimeError('Adjoint CG failed')
    if info>0:
        warnings.warn(f'Adjoint CG stopped without tolerance (info={info})',RuntimeWarning)
    if verbose:
        print('Adjoint CG residual:',np.linalg.norm(A@pI-b),'info:',info)
    p=np.zeros(len(mesh.nodes))
    p[I]=pI
    return p


def project_p1_to_p0(mesh,p):
    return np.mean(p[mesh.elements],axis=1)


def reduced_gradient(mesh,z,p,alpha=1e-4):
    """Inexact L2 gradient of f, with adjoint equation RHS (w-u)."""
    return alpha*z-project_p1_to_p0(mesh,p)


def compute_inexact_gradient(mesh,z,alpha=1e-4,w_value=-1.,
                             state_tol=1e-6,adjoint_tol=1e-6,verbose=False,
                             u0=None, p0=None, return_jacobian=False):
    if return_jacobian:
        u, J = solve_state(mesh,z,tol=state_tol,verbose=verbose,u0=u0,
                           return_jacobian=True)
    else:
        u = solve_state(mesh,z,tol=state_tol,verbose=verbose,u0=u0)
        J = None
    p=solve_adjoint(mesh,u,w_value=w_value,rtol=adjoint_tol,
                    verbose=verbose,J=J,p0=p0)
    g=reduced_gradient(mesh,z,p,alpha)
    return (u,p,g,J) if return_jacobian else (u,p,g)


def p0_l2_norm(mesh,z):
    areas,_=element_geometry(mesh)
    return float(np.sqrt(np.dot(areas,z*z)))


def p0_inner(mesh, a, b):
    areas, _ = element_geometry(mesh)
    return float(np.dot(areas * a, b))


def p0_norm(mesh, a):
    return np.sqrt(max(0.0, p0_inner(mesh, a, a)))


def value_f(mesh, z, u, alpha=1e-4, w=-1.0):
    areas, _ = element_geometry(mesh)
    bary, weights = triangle_quadrature()
    uq = u[mesh.elements] @ bary.T
    return (0.5 * np.sum(areas[:, None] * weights[None, :] * (uq-w)**2)
            + 0.5 * alpha * np.dot(areas, z*z))


def prox_phi(z, r, beta=1e-2):
    """L2 prox of beta||.||_1 + indicator_{[-25,25]}, r>0."""
    if r <= 0:
        raise ValueError('prox parameter must be positive')
    return np.clip(np.sign(z) * np.maximum(np.abs(z)-r*beta, 0.), -25., 25.)


def moreau_value(mesh, z, mu, beta=1e-2):
    v = prox_phi(z, mu, beta)
    areas, _ = element_geometry(mesh)
    return float(np.dot(areas, beta*np.abs(v) + 0.5/mu*(z-v)**2))


def moreau_grad(z, mu, beta=1e-2):
    return (z-prox_phi(z,mu,beta))/mu


def p0_transfers(mesh_coarse, mesh_fine):
    """L2 restriction=cell averaging and prolongation=R* for these meshes.

    Requires fine grid to have precisely twice the square divisions of coarse,
    and the diagonal convention in create_unit_square_mesh.
    """
    nc = int(round(np.sqrt(mesh_coarse.elements.shape[0]/2)))
    nf = int(round(np.sqrt(mesh_fine.elements.shape[0]/2)))
    if nf != 2*nc:
        raise ValueError('fine grid must be a single uniform refinement')
    cent = mesh_fine.nodes[mesh_fine.elements].mean(axis=1)
    scaled = cent*nc
    cell = np.minimum(np.floor(scaled).astype(int), nc-1)
    i,j=cell[:,0],cell[:,1]
    triangle = ((scaled[:,1]-j) > (scaled[:,0]-i)).astype(int)
    parent = 2*(i*nc+j)+triangle
    Nf=len(parent)
    Nc=mesh_coarse.elements.shape[0]
    R=sp.coo_matrix((np.full(Nf,.25),(parent,np.arange(Nf))),shape=(Nc,Nf)).tocsr()
    P=(4*R.T).tocsr()  # L2-adjoint, NOT ordinary Euclidean transpose
    if not (np.allclose(np.asarray(R.sum(axis=1)).ravel(),1.)
            and np.allclose(np.asarray(P.sum(axis=1)).ravel(),1.)):
        raise AssertionError('transfer not constant-preserving')
    return R,P


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


class ExactReducedHessian:
    """Matrix-free exact reduced L2 Hessian of the semilinear PDE objective.

    State:   -Delta u + u**3 = z.
    Adjoint: (-Delta + 3*u**2) p = desired_state-u.

    For direction s, solve
        A du = control_load(s),
        A dp = -mass_load((1 + 6*u*p)*du),
        H s = alpha*s - cell_average(dp).

    All products and nonlinear coefficients use the same 7-point triangle
    quadrature as the objective and PDE.  The factorization is reused for
    every Hessian-vector product at a fixed control.
    """
    def __init__(self, mesh, u, p, alpha, stats=None, J=None):
        self.stats = stats
        if self.stats is not None:
            self.stats['exact_hessian_builds'] += 1
        self.mesh = mesh
        self.alpha = float(alpha)
        self.areas, _ = element_geometry(mesh)
        self.E = mesh.elements
        self.I = mesh.interior
        self.bary, self.weights = triangle_quadrature()
        uq = u[self.E] @ self.bary.T
        pq = p[self.E] @ self.bary.T
        self.adjoint_weight = 1.0 + 6.0*uq*pq
        if J is None:
            _, J = assemble_nonlinear_terms(mesh, u)
        A = (get_stiffness(mesh) + J)[self.I][:, self.I].tocsc()
        self.solve_A = spla.factorized(A)

        # Conservative bound for this unit-square Dirichlet problem:
        # ||S'(z)||_(L2->L2) <= 1 by the Poincare inequality, hence
        # ||H|| <= alpha + ||1 + 6*u*p||_infinity.
        self.bound = self.alpha + float(np.max(np.abs(self.adjoint_weight)))
        # Cheap SPD preconditioner for Newton-CG.  A modest floor improves
        # the small-alpha PDE-controlled directions without changing H.
        self.preconditioner_diagonal = np.full(
            len(self.areas), max(self.alpha, 1e-3), dtype=float
        )

    def matvec(self, s):
        if self.stats is not None:
            self.stats['exact_hessian_products'] += 1
        s = np.asarray(s, dtype=float)
        du = np.zeros(len(self.mesh.nodes), dtype=float)
        du[self.I] = self.solve_A(assemble_control_load(self.mesh, s)[self.I])

        duq = du[self.E] @ self.bary.T
        bloc = -np.einsum(
            'eq,q,qi,e->ei',
            self.adjoint_weight*duq, self.weights, self.bary, self.areas
        )
        rhs = np.bincount(
            self.E.ravel(), weights=bloc.ravel(), minlength=len(du)
        )
        dp = np.zeros_like(du)
        dp[self.I] = self.solve_A(rhs[self.I])
        return self.alpha*s - project_p1_to_p0(self.mesh, dp)


def moreau_hessian_diagonal(z, mu, beta):
    """A generalized Hessian selection for MY(beta*|z|+I_box).

    At switching points the envelope is C1 but not necessarily C2;
    choose 1/mu as an admissible endpoint value.  For beta=0,
    z=0 is strictly inside the box and has ZERO Moreau curvature.
    """
    if mu <= 0 or beta < 0:
        raise ValueError('mu must be positive and beta nonnegative')
    a = np.abs(z)
    soft = (beta > 0.0) & (a <= mu*beta)
    box = a >= 25.0 + mu*beta
    return (soft | box).astype(float)/mu


class LBFGSHessian:
    """Limited-memory BFGS *Hessian* in the mesh-weighted L2 geometry.

    No dense matrix is formed. Each stored update has the form
        B <- B - (B s) (B s)^* / <s,Bs> + y y^* / <s,y>.
    The star uses the weighted L2 inner product.
    """
    def __init__(self, areas, gamma, pairs):
        self.areas = areas
        self.gamma = float(gamma)
        self.updates = []
        self.bound = self.gamma  # Conservative upper bound on ||B||_L2
        for s, y in pairs:
            Bs = self.matvec(s)
            sBs = self.inner(s, Bs)
            sy = self.inner(s, y)
            if sBs <= 0 or sy <= 0:
                continue
            self.updates.append((Bs, sBs, y, sy))
            # The negative BFGS rank-one term cannot increase lambda_max.
            self.bound += self.inner(y, y) / sy

    def inner(self, u, v):
        return float(np.dot(self.areas * u, v))

    def matvec(self, v):
        ans = self.gamma * np.asarray(v, dtype=float)
        for Bs, sBs, y, sy in self.updates:
            ans = ans - Bs * (self.inner(Bs, v) / sBs)
            ans = ans + y * (self.inner(y, v) / sy)
        return ans


class GalerkinHessian:
    """Coarse L2 Hessian Bc = R Bf P; R is weighted adjoint of P."""
    def __init__(self, fine_hessian, R, P):
        self.fine_hessian = fine_hessian
        self.R = R
        self.P = P
        self.bound = fine_hessian.bound
        if hasattr(fine_hessian, 'preconditioner_diagonal'):
            self.preconditioner_diagonal = np.asarray(
                self.R @ fine_hessian.preconditioner_diagonal
            ).ravel()

    def matvec(self, v):
        return self.R @ self.fine_hessian.matvec(self.P @ v)


class HessianPlusDiagonal:
    """For smoothed problems: L-BFGS model of f plus explicit Moreau curvature."""
    def __init__(self, hessian, diagonal):
        self.hessian = hessian
        self.diagonal = diagonal
        self.bound = hessian.bound + float(np.max(diagonal))
        if hasattr(hessian, 'preconditioner_diagonal'):
            self.preconditioner_diagonal = (
                hessian.preconditioner_diagonal + diagonal
            )

    def matvec(self, v):
        return self.hessian.matvec(v) + self.diagonal * v


class IRMTRMoreau:
    def __init__(self, meshes, config=None):
        self.cfg = config or Config()
        self.levels = [Level(mesh) for mesh in meshes]
        for ell in range(1, len(self.levels)):
            R, P = p0_transfers(meshes[ell-1], meshes[ell])
            self.levels[ell].R, self.levels[ell].P = R, P
        self.history = []
        self.lbfgs_pairs = {ell: [] for ell in range(len(self.levels))}
        self.lbfgs_updates = {ell: 0 for ell in range(len(self.levels))}
        self.counters = dict(eligible=0, attempted=0, used=0, fallback=0)
        self.cost = {
            ell:{'eval_calls':0,'eval_time':0.0,
                 'obj_calls':0,'obj_time':0.0,
                 'reused_trial_states':0,
                 'exact_hessian_builds':0,
                 'exact_hessian_products':0}
            for ell in range(len(self.levels))
            }


    def objective(self, ell, z, mu, return_state=False, u_initial=None):
        """Trial objective. Optionally return its converged PDE state for reuse.

        State reuse is safe because objective() and evaluate() use the same
        control, mesh, and state solver tolerance.
        """
        t0 = time.perf_counter()
        c = self.cfg
        u = None
        if not np.all(np.isfinite(z)):
            result = np.inf
        else:
            mesh = self.levels[ell].mesh
            if return_state:
                u, J = solve_state(mesh, z, tol=c.state_tol,
                                   u0=u_initial, return_jacobian=True)
            else:
                u = solve_state(mesh, z, tol=c.state_tol, u0=u_initial)
                J = None
            result = value_f(mesh, z, u, c.alpha, c.desired_state)
            if not c.smooth_only:
                result += moreau_value(mesh, z, mu, c.beta)
        self.cost[ell]['obj_calls'] += 1
        self.cost[ell]['obj_time'] += time.perf_counter()-t0
        return (result, (u, J)) if return_state else result

    def evaluate(self, ell, z, mu, tau, state=None, u_initial=None,
                 p_initial=None):
        """State+adjoint or adjoint-only with a cached converged state.

        The prescribed gradient-error budget is not a certified error bound.
        """
        t0 = time.perf_counter()
        c = self.cfg
        mesh = self.levels[ell].mesh
        if state is None:
            u, p, g, J = compute_inexact_gradient(
                mesh, z, alpha=c.alpha, w_value=c.desired_state,
                state_tol=c.state_tol, adjoint_tol=c.adjoint_tol,
                u0=u_initial,p0=p_initial, return_jacobian=True)
        else:
            if isinstance(state, tuple):
                u, J = state
            else:
                u, J = state, None
            p = solve_adjoint(mesh, u, w_value=c.desired_state,
                              rtol=c.adjoint_tol, J=J,p0=p_initial)
            g = reduced_gradient(mesh, z, p, c.alpha)
            self.cost[ell]['reused_trial_states'] += 1
        F=value_f(mesh,z,u,c.alpha,c.desired_state)#+moreau_value(mesh,z,mu,c.beta)
        if c.smooth_only:
            G= g.copy()
        else:
            F += moreau_value(mesh,z,mu,c.beta)
            G = g+moreau_grad(z,mu,c.beta)
        result = Evaluation(F,g,G,c.e_fraction*tau,u,p,J)
        self.cost[ell]['eval_calls']+=1
        self.cost[ell]['eval_time']+= time.perf_counter()-t0
        return result

    def norm(self,ell,z):
        return p0_norm(self.levels[ell].mesh,z)

    def model_diagonal(self, z, mu):
        """Positive diagonal B approximating curvature of f + Moreau(phi).

        Away from soft-threshold / box transitions, the Moreau curvature is
        either 1/mu or 0. At transition points we take 1/mu conservatively.
        """
        c = self.cfg
        if c.smooth_only:
            return np.full_like(z,c.curvature,dtype=float)
        else:
            return c.curvature + moreau_hessian_diagonal(z, mu, c.beta)

    def reset_lbfgs(self):
        """Reset across distinct Moreau smoothing stages, not on recursion."""
        self.lbfgs_pairs = {ell: [] for ell in range(len(self.levels))}
        self.lbfgs_updates = {ell: 0 for ell in range(len(self.levels))}

    def add_lbfgs_pair(self, ell, s, y):
        """Use the smooth reduced gradient difference (weighted L2 geometry)."""
        if self.cfg.hessian_mode != 'lbfgs':
            return False
        mesh = self.levels[ell].mesh
        sy = p0_inner(mesh, s, y)
        ns, ny = self.norm(ell, s), self.norm(ell, y)
        # Skip noisy, negative or very weak secant curvature.
        if not np.isfinite(sy) or ns <= 1e-14 or ny <= 1e-14:
            return False
        if sy <= 1e-8 * ns * ny:
            return False
        mem = self.lbfgs_pairs[ell]
        mem.append((s.copy(), y.copy()))
        if len(mem) > self.cfg.lbfgs_memory:
            del mem[:-self.cfg.lbfgs_memory]
        self.lbfgs_updates[ell] += 1
        return True

    def make_model(self, ell, z, mu, ev=None):
        """Build diagonal, L-BFGS, or exact reduced Hessian model.

        In exact mode, ev provides the already computed state and adjoint,
        avoiding repeated nonlinear state/adjoint solves at this control.
        """
        c = self.cfg
        if c.hessian_mode == 'diagonal':
            return self.model_diagonal(z, mu)
        if c.hessian_mode == 'lbfgs':
            areas, _ = element_geometry(self.levels[ell].mesh)
            H = LBFGSHessian(areas, c.curvature, self.lbfgs_pairs[ell])
        elif c.hessian_mode == 'exact':
            mesh = self.levels[ell].mesh
            if ev is not None and ev.u is not None and ev.p is not None:
                u, p = ev.u, ev.p
            else:
                u, p, _ = compute_inexact_gradient(
                    mesh, z, alpha=c.alpha, w_value=c.desired_state,
                    state_tol=c.state_tol, adjoint_tol=c.adjoint_tol
                )
            H = ExactReducedHessian(mesh, u, p, c.alpha,
                                    stats=self.cost[ell],
                                    J=(ev.J if ev is not None else None))
        else:
            raise ValueError('unknown hessian_mode')
        if c.smooth_only:
            return H
        D = moreau_hessian_diagonal(z, mu, c.beta)
        return HessianPlusDiagonal(H, D)

    @staticmethod
    def hessian_matvec(B, v):
        return B * v if isinstance(B, np.ndarray) else B.matvec(v)

    @staticmethod
    def hessian_bound(B):
        return float(np.max(B)) if isinstance(B, np.ndarray) else B.bound

    def tr_cg_step(self, ell, G, B, delta):
        """Preconditioned Steihaug CG in the weighted P0 L2 geometry.

        Preserves the original trust-region constraint and supports indefinite
        Hessians through negative-curvature boundary termination.  For an
        L-BFGS model without a preconditioner this reduces to ordinary CG.
        """
        inner = lambda u, v: p0_inner(self.levels[ell].mesh, u, v)
        s = np.zeros_like(G)
        r = G.copy()
        rnorm_initial = self.norm(ell, r)
        if rnorm_initial == 0 or delta <= 0:
            return s
        diag = getattr(B, 'preconditioner_diagonal', None)
        if diag is None:
            apply_Minv = lambda x: x.copy()
        else:
            diag = np.maximum(np.asarray(diag), 1e-12)
            apply_Minv = lambda x: x / diag
        zr = apply_Minv(r)
        rz = inner(r, zr)
        d = -zr

        def boundary_step(s, d):
            aa = inner(d, d)
            bb = 2.0 * inner(s, d)
            cc = inner(s, s) - delta**2
            disc = max(0.0, bb**2 - 4.0 * aa * cc)
            t = (-bb + np.sqrt(disc)) / (2.0 * aa)
            return s + t*d

        for _ in range(self.cfg.lbfgs_cg_maxiter):
            Bd = B.matvec(d)
            dBd = inner(d, Bd)
            if not np.isfinite(dBd) or dBd <= 0:
                return boundary_step(s, d)
            alpha = rz / dBd
            s_next = s + alpha*d
            if self.norm(ell, s_next) >= delta:
                return boundary_step(s, d)
            r_next = r + alpha*Bd
            if self.norm(ell, r_next) <= self.cfg.lbfgs_cg_rtol * rnorm_initial:
                return s_next
            zr_next = apply_Minv(r_next)
            rz_next = inner(r_next, zr_next)
            d = -zr_next + (rz_next/rz)*d
            s, r, rz = s_next, r_next, rz_next
        return s

    def cauchy_step(self, ell, G, B, delta, return_pred=False):
        """Cauchy safeguard. Reuse <G,BG> in its predicted reduction."""
        gnorm = self.norm(ell, G)
        if delta <= 0 or gnorm == 0:
            s = np.zeros_like(G)
            return (s, 0.0) if return_pred else s
        gBg = p0_inner(self.levels[ell].mesh, G, self.hessian_matvec(B, G))
        t = delta/gnorm
        if gBg > 0:
            t = min(t, gnorm*gnorm/gBg)
        s = -t*G
        predicted = t * gnorm**2 - .5*t*t*gBg
        return (s, predicted) if return_pred else s

    def model_step(self, ell, G, Bdiag, delta, return_pred=False):
        """Steihaug-CG/Cauchy step; avoid repeat exact Hessian products."""
        if delta <= 0:
            s = np.zeros_like(G)
            return (s, 0.0) if return_pred else s
        if not isinstance(Bdiag, np.ndarray):
            scg = self.tr_cg_step(ell, G, Bdiag, delta)
            # Preserve L-BFGS behavior; safeguard exact Newton-CG.
            if not hasattr(Bdiag, 'preconditioner_diagonal'):
                pred_cg = self.pred(ell, G, scg, Bdiag) if return_pred else None
                return (scg, pred_cg) if return_pred else scg
            scauchy, pred_cauchy = self.cauchy_step(
                ell, G, Bdiag, delta, return_pred=True)
            pred_cg = self.pred(ell, G, scg, Bdiag)
            if pred_cauchy > pred_cg:
                return (scauchy, pred_cauchy) if return_pred else scauchy
            return (scg, pred_cg) if return_pred else scg
        s = -G/Bdiag
        if self.norm(ell,s) > delta:
            lo, hi = 0., 1.
            while self.norm(ell,G/(Bdiag+hi)) > delta:
                hi *= 2.
            for _ in range(55):
                mid = .5*(lo+hi)
                if self.norm(ell,G/(Bdiag+mid)) > delta:
                    lo = mid
                else:
                    hi = mid
            s = -G/(Bdiag+hi)
        return (s, self.pred(ell,G,s,Bdiag)) if return_pred else s

    def pred(self, ell, G, s, Bdiag):
        mesh = self.levels[ell].mesh
        Bs = self.hessian_matvec(Bdiag, s)
        return -p0_inner(mesh,G,s)-.5*p0_inner(mesh,s,Bs)

    def fcd_ok(self, ell, G, s, delta, Bdiag, predicted=None):
        c = self.cfg
        gnorm = self.norm(ell, G)
        rhs = c.kappa_fcd*gnorm*min(gnorm/(1+self.hessian_bound(Bdiag)),delta)
        if predicted is None:
            predicted = self.pred(ell, G, s, Bdiag)
        return predicted+1e-12>=rhs

    def irmtr(self, ell, z_initial, mu, tau, inherited_delta=np.inf, B_initial=None, ev_initial=None):
        """Recursive uncorrected IRMTR: returns a CONTROL, not a step.

        The parent computes s_coarse = result.z - z_initial_coarse.
        The predicted decrease of a parent coarse step uses the INITIAL
        lower-level quadratic model (not the fine model).
        """
        c = self.cfg
        z0 = np.asarray(z_initial, dtype=float).copy()
        z = z0.copy()
        delta = min(c.delta_local, inherited_delta)
        limit = c.max_inner if ell == len(self.levels)-1 else c.max_coarse_inner
        ev_cached = ev_initial
        accepted_trial_state = None
        B_cached = None
        coarse_ev_cached = None
        last_coarse_state = None
        last_coarse_adjoint = None
        last_fine_adjoint = None
        pending_pair = None  # (accepted step, smooth gradient at old control)
        single_step_model_prediction = None
        for k in range(limit):
            if ev_cached is None:
                ev_cached = self.evaluate(ell,z,mu,tau,
                                          state=accepted_trial_state,
                                          p_initial=last_fine_adjoint)
                accepted_trial_state = None
            ev = ev_cached
            # The new fine gradient is needed anyway for stopping. Use it now
            # to record the previous accepted step's secant pair at zero extra solves.
            if pending_pair is not None:
                old_step, old_g = pending_pair
                self.add_lbfgs_pair(ell, old_step, ev.g - old_g)
                pending_pair = None
            #ev = self.evaluate(ell, z, mu, tau)
            #if k==0 and ev_initial is not None:
            #    ev = ev_initial
            #else:
            #    ev = self.evaluate(ell, z, mu, tau)
            stationarity = self.norm(ell, ev.G) + ev.e
            if (c.verbose and c.progress_every > 0
                and ell == len(self.levels)-1
                and (k == 0 or (k + 1) % c.progress_every == 0)):
                print(f'    mu={mu:.3e} inner k={k:4d} '
                      f'F_mu={ev.F:.9e} ||G_mu||+e={stationarity:.3e} '
                      f'delta={delta:.3e}')
            if stationarity <= tau:
                if c.verbose and ell == len(self.levels)-1:
                    print(f'  Inner converged: ||G||+e={stationarity:.6e} '
                          f'<= tau={tau:.6e}, iterations={k}')
                return InnerResult(z, k, stationarity, True, 'stationarity')
            if np.isfinite(inherited_delta):
                remaining = inherited_delta - self.norm(ell, z-z0)
                if remaining <= c.tau_delta*inherited_delta:
                    return InnerResult(z, k, stationarity, False, 'parent radius limit')
                delta = min(delta, remaining)
            if delta <= 1e-14:
                return InnerResult(z, k, stationarity, False, 'radius underflow')
            
            if k==0 and B_initial is not None:
                Bdiag = (B_initial.copy() if isinstance(B_initial, np.ndarray)
                         else B_initial)
            elif B_cached is not None:
                Bdiag = B_cached
            else:
                Bdiag = self.make_model(ell, z, mu, ev=ev)

            

            #step = self.model_step(ell, ev.G, Bdiag, delta)
            #predicted = self.pred(ell, ev.G, step, Bdiag)
            step = None
            predicted = None
            mode = 'current'
            zeta = np.nan
            eligible = False
            reject_reason = ''
            force_fine = (c.smooth_only and ell== len(self.levels)-1 and (k+1)%5==0)

            if ell > 0 and (not c.smooth_only or ell==len(self.levels)-1) and not force_fine: #and mu>=0.05:
                
                R, P = self.levels[ell].R, self.levels[ell].P
                zc = R @ z
                if coarse_ev_cached is None:
                    coarse_ev_cached = self.evaluate(
                        ell-1, zc, mu, tau,
                        u_initial=last_coarse_state,
                        p_initial=last_coarse_adjoint)
                cev = coarse_ev_cached
                zeta = self.norm(ell-1, cev.G - R @ ev.G)
                if c.smooth_only and ell == len(self.levels)-1:
                    Gf = ev.G
                    Gc = cev.G
                    RGf = R @ Gf
                    consistency_error = self.norm(ell-1,Gc-RGf)
                    eta_cons = consistency_error/max(self.norm(ell-1,RGf), 1e-14)
                    G_projected = P@RGf
                    G_fine_only = Gf -G_projected
                    eta_fine = self.norm(ell,G_fine_only)/max(self.norm(ell,Gf),1e-14)
                    if k<3 or k%100 ==0:
                        print(
                            f"k={k:4d},"
                            f"eta_cons={eta_cons:.6e},"
                            f"eta_fine={eta_fine:.6e},"
                            f"zeta={zeta:.6e},"
                            f"||Gf||={self.norm(ell,Gf):.6e}")
                #coarsening_error = self.norm(ell,z-P@zc)
                
                #if ell == len(self.levels)-1 and mu <= 0.025:
                #    if k < 3 or k % 50 == 0:
                #        print(f"mu={mu:.5f},k={k}," 
                #              f"||z-PRz|| = {coarsening_error:.6e},"
                #              f"bound = {coarsening_error/mu:.6e}")

                #        d_smooth = cev.g - R @ ev.g

                #        d_moreau = (moreau_grad(zc, mu, c.beta)- R @ moreau_grad(z, mu, c.beta))

                #        print(f"mu={mu:.5f}, k={k}, "f"zeta={zeta:.3e}, "f"smooth_gap={self.norm(ell-1, d_smooth):.3e}, "f"moreau_gap={self.norm(ell-1, d_moreau):.3e}")
                gf = self.norm(ell,ev.G)
                gc = self.norm(ell-1,cev.G)
                ml_threshold = c.kappa_ml*min(delta,gf)
                et_threshold = c.kappa_et*gf
                ml_ok = zeta <= ml_threshold
                et_ok = gc >= et_threshold
                eligible = ml_ok and et_ok

                #eligible = (zeta <= c.kappa_ml * min(delta, self.norm(ell, ev.G))
                #            and self.norm(ell-1, cev.G) >= c.kappa_et*self.norm(ell, ev.G))
                #if ell== len(self.levels)-1 and mu<= 0.025:
                #    if k < 3 or k % 50 == 0:
                #        print(
                #            f"mu={mu:.5f}, k={k}, "
                #            f"zeta={zeta:.3e}, "
                #            f"ML threshold={ml_threshold:.3e}, "
                #            f"ML condition={ml_ok}, "
                #            f"coarse gradient={gc:.3e}, "
                #            f"ET threshold={et_threshold:.3e}, "
                #            f"ET condition={et_ok}"
                #            )
                if eligible:
                    self.counters['eligible'] += 1
                    # For nested uniform P0 elements, ||P||_{L2->L2}=1.
                    inherited = delta
                    self.counters['attempted'] += 1
                    Bc = (R @ Bdiag if isinstance(Bdiag, np.ndarray)
                          else GalerkinHessian(Bdiag, R, P))
                    low = self.irmtr(ell-1, zc, mu, tau, inherited_delta=inherited, B_initial=Bc, ev_initial = cev)
                    sc = low.z - zc
                    sf = P @ sc
                    
                    #Bc = self.model_diagonal(zc, mu)
                    #Bc = R @ Bdiag
                    # For a single-step child, its own model prediction is
                    # already known. That model is exactly Bc = R Bf P at zc.
                    # Reusing it avoids another fine-grid Hessian action.
                    if low.local_predicted is not None:
                        coarse_pred = low.local_predicted
                    elif not np.any(sc):
                        coarse_pred = 0.0
                    else:
                        coarse_pred = self.pred(ell-1, cev.G, sc, Bc)
                    # Exact Galerkin identity: Bc = R Bf P, R = P^* in L2.
                    # Same fine quadratic term, so no second fine H*sf action:
                    fine_pred = coarse_pred + p0_inner(
                        self.levels[ell-1].mesh, cev.G - R @ ev.G, sc)
                    #current_step = self.model_step(ell,ev.G,Bdiag,delta)
                    #current_pred = self.pred(ell,ev.G,current_step,Bdiag)
                    #step_efficiency = (fine_pred/current_pred if current_pred>1e-14 else np.nan)
                    
                    #if coarse_pred > 1e-14:
                    #    model_ratio = fine_pred/coarse_pred
                    #    discrepancy_ratio = (zeta*self.norm(ell-1,sc)/coarse_pred)
                    #else:
                    #    model_ratio = np.nan
                    #    discrepancy_ratio=np.nan
                        
                        
                    
                        
                        
                    # Print model-quality diagnostics at the finest level
                    #if ell == len(self.levels) - 1:
                    #    if k < 3 or k % 20 == 0:
                            
                    #        print(f"mu={mu:.5f}, k={k},"
                    #                  f"coarse_model_ratio = {model_ratio:.4f},"
                    #                  f"step_efficiency={step_efficiency:.4f}")
                    #        print(
                    #            f"mu={mu:.5f}, k={k}, "
                    #            f"D_coarse={coarse_pred:.6e}, "
                    #            f"D_fine={fine_pred:.6e}, "
                    #            f"model_ratio={model_ratio:.4f}, "
                    #            f"discrepancy_ratio={discrepancy_ratio:.4f}"
                    #        )
                    #discrepancy = abs(fine_pred - coarse_pred)
                    #relative_discrepancy = discrepancy/max(abs(coarse_pred),1e-14)
                    
                   # print(f"level={ell}," f"zeta={zeta:.3e}, "f"step_norm={self.norm(ell-1, sc):.3e}," 
                   #       f"coarse_pred={coarse_pred:.3e},"
                   #       f"fine_pred={fine_pred:.3e},"
                   #       f"relative model discrepancy = {relative_discrepancy:.3e}")
                    feasible = self.norm(ell, sf) <= delta*(1+1e-10)
                    fcd = self.fcd_ok(ell-1, cev.G, sc, inherited, Bc,
                                      predicted=coarse_pred)
                    model_quality_ok=(coarse_pred>1e-14 and fine_pred >= 0.5*coarse_pred)
                    #efficiency_ok = (current_pred > 1e-14 and fine_pred >= 0.9 * current_pred)
                    if feasible and model_quality_ok and fcd :#and efficiency_ok:
                        step, predicted, mode = sf, coarse_pred, 'coarse'
                        self.counters['used'] += 1
                    else:
                        self.counters['fallback'] += 1
                        reject_reason = ('infeasible' if not feasible else
                                         'nonpositive coarse prediction' if coarse_pred <= 1e-14 else
                                         'coarse FCD failed' if not fcd else
                                         'fine-model quality failed')
            if mode == 'current':
                step, predicted = self.model_step(
                    ell, ev.G, Bdiag, delta, return_pred=True)
            if mode == 'current' and not self.fcd_ok(
                    ell, ev.G, step, delta, Bdiag, predicted=predicted):
                return InnerResult(z, k, stationarity, False, 'FCD failed')
            if predicted <= 0 or not np.isfinite(predicted):
                return InnerResult(z, k, stationarity, False, 'nonpositive predicted reduction')
            step_ratio = self.norm(ell,step)/max(delta,1e-14)
            if (c.smooth_only and c.hessian_mode == 'diagonal'
                and ell==len(self.levels)-1 and mode == "coarse"
                and (k<3 or k%200==0)):
                fine_step=self.model_step(ell,ev.G,Bdiag,delta)
                D_fine_opt = self.pred(ell,ev.G,fine_step, Bdiag)
                D_coarse_actual = self.pred(ell,ev.G, step, Bdiag)
                eta_step = (D_coarse_actual/D_fine_opt if D_fine_opt>1e-14 else np.nan)
                print(f"\nk={k}, actual eta_step={eta_step:.4f}")
                for j in range(ell-1,-1,-1):
                    gproj = ev.G.copy()
                    for lev in range(ell,j,-1):
                        gproj = self.levels[lev].R @ gproj
                    for lev in range(j+1,ell+1):
                        gproj = self.levels[lev].P @ gproj
                    s_best = -gproj/c.curvature
                    step_norm = self.norm(ell, s_best)
                    if step_norm > delta:
                        s_best *= delta/step_norm
                    D_best = self.pred(ell,ev.G,s_best,Bdiag)
                    efficiency_bound = D_best / D_fine_opt
                    fine_only = self.norm(ell,ev.G-gproj)/max(self.norm(ell,ev.G),1e-14)
                    print(
                        f" level {j}:"
                        f"best_efficiency={efficiency_bound:.4f},"
                        f"fine_only={fine_only:.4f}"
                        )
                print(
                    f"k={k:4d},"
                    f"eta_step={eta_step:.6f},"
                    f"D_fine_opt={D_fine_opt:.6e},"
                    f"D_coarse_actual={D_coarse_actual:.6e}"
                    )
            trial = z + step
            F_trial, trial_state = self.objective(
                ell, trial, mu, return_state=True, u_initial=ev.u)
            cred = ev.F - F_trial
            rho = cred / predicted
            accepted = bool(rho >= c.eta1)
            if limit == 1 and ell < len(self.levels)-1:
                # Parent's coarse_pred is the quadratic reduction in THIS
                # level's original model (not the deeper model's prediction).
                single_step_model_prediction = (
                    (fine_pred if mode == 'coarse' else predicted)
                    if accepted else 0.0)
            #delta_before = delta
            
            if accepted:
                # Defer the gradient difference to the next loop iteration.
                # Rejected steps do NOT contribute L-BFGS secant pairs.
                if c.hessian_mode == 'lbfgs':
                    pending_pair = (step.copy(), ev.g.copy())
                last_fine_adjoint = ev.p
                z = trial
                ev_cached = None
                accepted_trial_state = trial_state
                B_cached = None
                if coarse_ev_cached is not None:
                    last_coarse_state = coarse_ev_cached.u
                    last_coarse_adjoint = coarse_ev_cached.p
                coarse_ev_cached = None
            else:
                if not (k==0 and B_initial is not None):
                    B_cached = Bdiag
                
            #if accepted and ell==len(self.levels)-1:
            #    ev_after = self.evaluate(ell,z,mu,tau)
            #    stationarity_after = self.norm(ell,ev_after.G)+ev_after.e
            #    self.history_stationarity = getattr(self, 'history_stationarity',[])
            #    self.history_stationarity.append({'mu':mu,'mode':mode,'before':stationarity,'after':stationarity_after,'change':stationarity_after-stationarity})
            if rho >= c.eta2:
                delta = min(c.gamma_expand*delta, c.delta_max)
            elif rho < c.eta1:
                delta = c.gamma_shrink*delta
            #if np.isfinite(inherited_delta):
            #    delta = min(delta, max(0., inherited_delta-self.norm(ell, z-z0)))
            #print(f"level={ell}, k={k}, "f"rho={rho:.6e}, "f"delta_before={delta_before:.6e}, "f"delta_after={delta:.6e}")
            self.history.append(dict(level=ell,k=k,mode=mode,mu=mu,F=ev.F,
                                     residual=stationarity,delta=delta,rho=rho,
                                     accepted=accepted,cred=cred,predicted=predicted,
                                     zeta=zeta,eligible=eligible,
                                     rejection=reject_reason,step_ratio = step_ratio))
            #if c.verbose  and (k < 3 or k % 50 == 0): #and ell == len(self.levels)-1
            #    indent = " "*(2-ell)
            #    print(
            #        f"{indent}Level {ell}, "
            #        f"k={k:04d}, "
            #        f"mode={mode}, "
            #        f"F={ev.F:.6e}, "
            #        f"rho={rho:.3e}, "
            #        f"delta={delta:.3e}, "
            #        f"zeta={zeta:.3e}"
            #    )
                #print(f'  inner k={k:04d} level={ell} {mode:7s} '
                #      f'F={ev.F:.7e} stat<={stationarity:.3e} '
                #      f'rho={rho:.3e} delta={delta:.2e} '
                #      f'zeta={zeta:.2e}')
        #ev = ev_cached if ev_cached is not None else self.evaluate(ell, z, mu, tau)
        
        if ell<len(self.levels)-1:
            return InnerResult(z, limit, np.nan, False,
                               'coarse iteration limit',
                               local_predicted=single_step_model_prediction)
        ev = (ev_cached if ev_cached is not None else self.evaluate(ell,z,mu,tau))
        final_stat = self.norm(ell, ev.G) + ev.e
        if c.verbose:# and ell == len(self.levels)-1:
            print(f'  Inner reached max_inner: stationarity={final_stat:.6e}, tau={tau:.6e}')
        return InnerResult(z, limit, final_stat, False, 'max_inner reached')

    def original_stationarity(self,z,r,tau):
        """(1/r)||z - prox_{r phi}(z-r*g)|| + prescribed e."""
        ell=len(self.levels)-1
        # For outer test, just evaluate gradient f; mu choice does not change g.
        ev=self.evaluate(ell,z,1.,tau)
        step=prox_phi(z-r*ev.g,r,self.cfg.beta)
        psi=self.norm(ell,z-step)/r
        return psi+ev.e,psi,ev.e

    def solve_outer(self,z0,mu0,tau_fixed,eps,
                    theta_mu=.5,r=1.,max_outer=None):
        """Moreau-Yosida continuation with mu and tau both decreasing."""
        if not (mu0>0 and eps>0 and 0<tau_fixed<eps and r>0
                and 0<theta_mu<1):
            raise ValueError('require mu0>0, eps>tau_fixed>0, r>0, 0<theta_mu<1')
        c=self.cfg
        h=len(self.levels)-1
        z=np.asarray(z0,dtype=float).copy()
        mu=mu0
        tau=tau_fixed
        history=[]
        nouter=c.max_outer if max_outer is None else max_outer
        for j in range(nouter):
            if c.verbose:
                print(f'outer j={j}, mu={mu:.3e}, tau={tau:.3e}')
            z_start = z.copy()
            if c.hessian_mode == 'lbfgs':
                self.reset_lbfgs()  # Do not reuse secant pairs across mu stages.
            low=self.irmtr(h,z,mu,tau,inherited_delta=np.inf)
            z=low.z
            z_feas = np.clip(z,-25.0,25.0)
            feas_upper, feas_psi, feas_e = self.original_stationarity(z_feas,r,tau)
            print(f"Feasible residual + e: {feas_upper:.9e}, "
                  f"box violation: {np.max(np.maximum(np.abs(z_feas)-25., 0.)):.3e}")
            # Fixed tau gives the same prescribed error allowance at every stage.
            stationarity,psi,e=self.original_stationarity(z,r,tau)
            history.append(dict(j=j,mu=mu,tau=tau,F_mu=self.objective(h,z,mu),
                                inner_converged=low.converged,
                                inner_reason=low.reason,
                                psi=psi,e=e,stationarity_upper=stationarity,
                                z_start=z_start.copy(),
                                z_end = z.copy(),
                                inner_iterations=low.iterations,))
            if c.verbose:
                print(f'  outer prox residual={psi:.3e}, '
                      f'prescribed e={e:.3e}, total={stationarity:.3e}')
            if not low.converged:
                #if c.verbose:
                #    print(f'  Inner IRMTR did not converge: {low.reason}; '
                #          'stopping outer smoothing without reducing mu.')
                return OuterResult(z_feas,history,False)
            if feas_upper<=eps:
                return OuterResult(z_feas,history,True)
            mu*=theta_mu
            tau*=theta_mu
        return OuterResult(np.clip(z,-25,25),history,False)




def plot_results(mesh, solver, out, eps=1e-3):
    """Display results in Spyder Plots pane; do not save files."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from collections import Counter
    triang = mtri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh.elements)
    u = solve_state(mesh, out.z, tol=solver.cfg.state_tol)
    fig, axs = plt.subplots(1,2,figsize=(11,4.8))
    a=axs[0].tripcolor(triang,facecolors=out.z,shading='flat',cmap='coolwarm')
    fig.colorbar(a,ax=axs[0],label='Control z')
    axs[0].set_title('Finest-level control')
    b=axs[1].tripcolor(triang,u,shading='gouraud',cmap='viridis')
    fig.colorbar(b,ax=axs[1],label='State u')
    axs[1].set_title('Finest-level PDE state')
    for ax in axs: ax.set_aspect('equal');ax.set_xlabel('x');ax.set_ylabel('y')
    fig.tight_layout()

    hist=[r for r in solver.history if r['level']==len(solver.levels)-1]
    if hist:
        fig2,axes=plt.subplots(2,2,figsize=(11,7))
        x=np.arange(len(hist))
        axes[0,0].plot(x,[r['F'] for r in hist]);axes[0,0].set_title('Finest-level smoothed objective')
        axes[0,1].semilogy(x,[max(r['residual'],1e-16) for r in hist]);axes[0,1].set_title('Finest-level stationarity')
        axes[1,0].plot(x,[r['rho'] for r in hist]);axes[1,0].set_title('Acceptance ratio')
        axes[1,1].semilogy(x,[max(r['delta'],1e-16) for r in hist]);axes[1,1].set_title('Trust-region radius')
        for ax in axes.flat:ax.grid(True,alpha=.25);ax.set_xlabel('Finest-level iteration')
        fig2.tight_layout()
    if out.history:
        fig3,ax=plt.subplots(figsize=(6,4))
        ax.semilogy([r['j'] for r in out.history],
                    [r['stationarity_upper'] for r in out.history],'o-')
        ax.axhline(eps,color='gray',linestyle='--',label='epsilon')
        ax.set_title('Outer proximal stationarity');ax.set_xlabel('Smoothing stage')
        ax.grid(True,alpha=.25);ax.legend();fig3.tight_layout()
    counts=Counter((r['level'],r['mode']) for r in solver.history)
    print('TR steps by level and mode:',dict(counts))
    plt.show()

import time
from collections import Counter

def check_exact_hessian(mesh, cfg, z, mu, epsilon=1e-4):
    """Check generalized Hessian on a smooth region (no MY switches)."""
    rng = np.random.default_rng(2026)
    solver = IRMTRMoreau([mesh], cfg)
    ev = solver.evaluate(0, z, mu, tau=1e-5)
    H = solver.make_model(0, z, mu, ev=ev)
    s = rng.standard_normal(len(z))
    s /= solver.norm(0, s)
    t = rng.standard_normal(len(z))
    t /= solver.norm(0, t)
    ev_plus = solver.evaluate(0, z+epsilon*s, mu, tau=1e-5)
    ev_minus = solver.evaluate(0, z-epsilon*s, mu, tau=1e-5)
    finite_diff = (ev_plus.G - ev_minus.G) / (2.0 * epsilon)
    Hs, Ht = H.matvec(s), H.matvec(t)
    fd_err = solver.norm(0, finite_diff-Hs) / max(solver.norm(0, Hs), 1e-14)
    lhs = p0_inner(mesh, s, Ht)
    rhs = p0_inner(mesh, t, Hs)
    symmetry_err = abs(lhs-rhs)/max(abs(lhs), abs(rhs), 1e-14)
    print(f'  Hessian finite-difference relative error: {fd_err:.3e}')
    print(f'  Weighted-L2 symmetry relative error:     {symmetry_err:.3e}')
    return fd_err, symmetry_err


def check_moreau_curvature(mu, beta):
    """Finite-difference test of MY generalized Hessian away from switches."""
    if beta > 0.0:
        points = np.array([0.0, .5*mu*beta, 2.0*mu*beta,
                           25.0 + .5*mu*beta, 25.0 + 2.0*mu*beta,
                           -25.0 - 2.0*mu*beta])
        epsilon = min(1e-6, mu*beta/10.0)
    else:
        points = np.array([0.0, 1.0, 24.0, 26.0, -26.0])
        epsilon = 1e-6
    fd = (moreau_grad(points+epsilon, mu, beta)
          -moreau_grad(points-epsilon, mu, beta))/(2*epsilon)
    model = moreau_hessian_diagonal(points, mu, beta)
    difference = np.max(np.abs(fd - model))
    print(f'  Moreau diagonal finite-difference abs error: {difference:.3e}')
    print(f'  Moreau diagonal selected values: {model}')
    return difference


def report_solution(name, solver, out, wall, original_r, eps):
    """Report the actual nonsmooth objective and original proximal residual."""
    finest_level = len(solver.levels)-1
    mesh = solver.levels[-1].mesh
    z = out.z
    u = solve_state(mesh, z, tol=solver.cfg.state_tol)
    value_smooth = value_f(mesh, z, u, solver.cfg.alpha, solver.cfg.desired_state)
    areas, _ = element_geometry(mesh)
    box_violation = float(np.max(np.maximum(np.abs(z)-25.0, 0.0)))
    l1_term = solver.cfg.beta * np.dot(areas, np.abs(z))
    original_value = (value_smooth + l1_term if box_violation < 1e-12 else np.inf)
    # This is an independent final evaluation, not a claim of verified inexactness.
    tau_final = out.history[-1]["tau"]
    stationarity, psi, e = solver.original_stationarity(z, original_r, tau_final)
    histories = [row for row in solver.history if row['level'] == finest_level]
    accepted_coarse = sum(row['mode']=='coarse' and row['accepted'] for row in histories)
    rejections = sum(not row['accepted'] for row in histories)
    z_feas = np.clip(z, -25.0, 25.0)

    u_feas = solve_state(mesh, z_feas,tol=solver.cfg.state_tol)

    F_feas = value_f(mesh, z_feas, u_feas,solver.cfg.alpha,solver.cfg.desired_state)

    F_feas += solver.cfg.beta * np.dot(areas, np.abs(z_feas))

    tau_final = out.history[-1]["tau"]

    stat_feas, psi_feas, e_feas = solver.original_stationarity(z_feas, original_r, tau_final)

    print(f"Feasible original objective: {F_feas:.12e}")
    print(f"Projected-control proximal residual: {psi_feas:.9e}")
    print(f"Projected-control stationarity + e: {stat_feas:.9e}") 
    print(f'\n========== {name}: RESULT ==========')
    print('Outer converged:', out.converged)
    print(f'Elapsed wall time: {wall:.3f} seconds')
    print('Fine trust-region steps:', len(histories))
    print('Accepted coarse corrections:', accepted_coarse)
    print('Rejected fine steps:', rejections)
    print(f'Original proximal residual: {psi:.9e}')
    print(f'Original residual + prescribed error: {stationarity:.9e}')
    print(f'Box violation (max): {box_violation:.9e}')
    print(f'Original nonsmooth objective (inf if infeasible): {original_value:.12e}')
    print(f'Smooth f(z): {value_smooth:.12e}  L1 penalty: {l1_term:.12e}')
    print('Coarse eligibility / attempts / used / fallback:', solver.counters)
    print('\nStage | mu       | tau      | inner steps | inner ok | prox residual | reason')
    for stage in out.history:
        print(f"{stage['j']:5d} | {stage['mu']:.2e} | {stage['tau']:.2e} |"
              f" {stage['inner_iterations']:11d} | {str(stage['inner_converged']):>8s} |"
              f" {stage['psi']:.5e} | {stage['inner_reason']}")
    if not out.converged:
        print('WARNING: Full original-problem stopping criterion was not met.')
    return dict(name=name, time=wall, iterations=len(histories),
                converged=out.converged, prox=psi, original_F=original_value,
                box_violation=box_violation)


def main():
    parser = argparse.ArgumentParser(
        description='FULL nonsmooth semilinear PDE control: exact Hessian + MY continuation'
    )
    parser.add_argument('--base-n', type=int, default=8,
                        help='4 grids: n,2n,4n,8n (default 8,16,32,64)')
    parser.add_argument('--full', action='store_true',
                        help='Very large meshes (64,128,256,512); costly')
    parser.add_argument('--solver', choices=['both','single','multi'], default='both')
    parser.add_argument('--hessian-mode', choices=['exact','lbfgs','diagonal'], default='exact')
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--alpha', type=float, default=1e-4)
    parser.add_argument('--box', type=float, default=25.0,
                        help='Current implementation uses [-25,25]; other values unsupported')
    parser.add_argument('--mu0', type=float, default=0.4)
    parser.add_argument('--tau-fixed', '--tau0', dest='tau_fixed',
                        type=float, default=8e-5,
                        help='Initial inner tolerance; decreases with theta-mu; must be < eps. '
                             '--tau0 accepted as a backward-compatible alias')
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--theta-mu', type=float, default=0.5,
                        help='Decrease mu and tau each outer stage')
    parser.add_argument('--prox-r', type=float, default=1.0)
    parser.add_argument('--max-outer', type=int, default=8)
    parser.add_argument('--max-inner', type=int, default=4000)
    parser.add_argument('--max-coarse-inner', type=int, default=1)
    parser.add_argument('--cg-maxiter', type=int, default=50)
    parser.add_argument('--cg-rtol', type=float, default=1e-3)
    parser.add_argument('--kappa-ml', type=float, default=0.15)
    parser.add_argument('--kappa-et', type=float, default=0.2)
    parser.add_argument('--progress-every', type=int, default=25)
    parser.add_argument('--lbfgs-memory', type=int, default=5)
    parser.add_argument('--fixed-mu', action='store_true',
                        help='Optional diagnostic: one fixed-mu full MY subproblem, no outer loop')
    parser.add_argument('--smooth-only', action='store_true',
                        help='Optional diagnostic ONLY: omit MY; default includes it')
    parser.add_argument('--hessian-check', action='store_true',
                        help='Small-mesh Hessian-vector finite-difference/symmetry check')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--plots', action='store_true', help='Display Spyder/Matplotlib plots')
    args, _unknown = parser.parse_known_args()  # compatible with Spyder runfile
    if args.box != 25.0:
        parser.error('This script currently implements the fixed box [-25,25].')
    if args.smooth_only and not args.fixed_mu:
        parser.error('--smooth-only is a diagnostic for --fixed-mu; full mode includes MY.')
    if args.beta < 0 or args.alpha <= 0:
        parser.error('beta must be nonnegative and alpha positive')
    if args.cg_maxiter <= 0 or args.max_inner <= 0 or args.max_outer <= 0:
        parser.error('Iteration limits must be positive')
    if not (args.eps > args.tau_fixed > 0):
        parser.error('Require 0 < --tau-fixed < --eps (e.g. 1e-4 < 1e-3)')
    ns = (64,128,256,512) if args.full else tuple(args.base_n * 2**i for i in range(4))
    meshes = [create_unit_square_mesh(n) for n in ns]
    finest = meshes[-1]
    print('Mesh n:', ns)
    print('Control DOFs:', [len(m.elements) for m in meshes])
    print('\n========== FULL MOREAU-YOSIDA MODEL ==========' if not args.smooth_only
          else '\n========== SMOOTH-ONLY DIAGNOSTIC ==========')
    print('Objective: f(z) + M_mu[beta*||z||_1 + I_{[-25,25]}](z)')
    print('PDE reduced Hessian:', args.hessian_mode)
    print('Moreau regularization: enabled' if not args.smooth_only else
          'Moreau regularization: DISABLED (diagnostic)')
    print(f'beta={args.beta}, alpha={args.alpha}')
    print(f'outer mu0={args.mu0}, tau0={args.tau_fixed}, eps={args.eps}, max_outer={args.max_outer}')

    def make_config():
        return Config(alpha=args.alpha, beta=args.beta,
                      desired_state=-1.0,
                      smooth_only=args.smooth_only,
                      hessian_mode=args.hessian_mode,
                      lbfgs_memory=args.lbfgs_memory,
                      lbfgs_cg_maxiter=args.cg_maxiter,
                      lbfgs_cg_rtol=args.cg_rtol,
                      kappa_ml=args.kappa_ml,
                      kappa_et=args.kappa_et,
                      max_inner=args.max_inner,
                      max_coarse_inner=args.max_coarse_inner,
                      max_outer=args.max_outer,
                      progress_every=args.progress_every,
                      verbose=not args.quiet)

    if args.hessian_check and args.hessian_mode == 'exact':
        print('\n========== EXACT + MY HESSIAN CHECK (SMOOTH REGIONS) ==========')
        test_mesh = meshes[0]
        test_z = 2.0 * np.ones(len(test_mesh.elements))
        # With beta>0, soft region is |z| < mu*beta, box region is |z|>25+mu*beta;
        # at z=2 we are away from either switching surface.
        check_exact_hessian(test_mesh, make_config(), test_z, args.mu0)
        if not args.smooth_only:
            check_moreau_curvature(args.mu0, args.beta)

    z0 = 2.0*np.ones(len(finest.elements))
    solvers = []
    if args.solver in ('both','single'):
        solvers.append(('Single-level TR', [finest]))
    if args.solver in ('both','multi'):
        solvers.append(('Multilevel IRMTR', meshes))
    results=[]
    for name, selected_meshes in solvers:
        solver = IRMTRMoreau(selected_meshes, make_config())
        h = len(solver.levels)-1
        print(f'\n========== {name.upper()} ==========')
        start = time.perf_counter()
        if args.fixed_mu:
            inner = solver.irmtr(h, z0.copy(), args.mu0, args.tau_fixed)
            elapsed = time.perf_counter()-start
            ev = solver.evaluate(h, inner.z, args.mu0, args.tau_fixed)
            total, psi, e = solver.original_stationarity(inner.z, args.prox_r, args.tau_fixed)
            rows=[row for row in solver.history if row['level']==h]
            print(f'Fixed mu={args.mu0}, tau={args.tau_fixed} (NO outer continuation)')
            print('Inner converged:', inner.converged, 'reason:', inner.reason)
            print('Inner iterations:', inner.iterations)
            print('Smoothed objective:', f'{ev.F:.12e}')
            print('Smoothed gradient norm:', f'{solver.norm(h, ev.G):.9e}')
            print('Original proximal residual:', f'{psi:.9e}')
            print('Accepted coarse corrections:',
                  sum(row['mode']=='coarse' and row['accepted'] for row in rows))
            print('Rejected steps:', sum(not row['accepted'] for row in rows))
            print('Elapsed seconds:', f'{elapsed:.3f}')
            results.append(dict(name=name, time=elapsed, iterations=inner.iterations,
                                converged=inner.converged, prox=psi))
        else:
            out=solver.solve_outer(z0.copy(), mu0=args.mu0, tau_fixed=args.tau_fixed,
                                   eps=args.eps, theta_mu=args.theta_mu,
                                   r=args.prox_r, max_outer=args.max_outer)
            elapsed=time.perf_counter()-start
            results.append(report_solution(name, solver, out, elapsed,
                                           args.prox_r, args.eps))
            if args.plots:
                plot_results(finest, solver, out, eps=args.eps)
            print('PDE and exact-Hessian work by level:')
            for level, work in solver.cost.items():
                print(f"  level {level}: evaluations={work['eval_calls']}, "
                      f"trial objectives={work['obj_calls']}, "
                      f"reused trial states={work['reused_trial_states']}, "
                      f"Hessian builds={work['exact_hessian_builds']}, "
                      f"Hessian-vector products={work['exact_hessian_products']}")
    print('\n========== COMPARISON ==========')
    for r in results:
        print(f"{r['name']}: iterations={r['iterations']}, "
              f"time={r['time']:.3f}s, converged={r['converged']}, "
              f"original_prox={r['prox']:.3e}")


if __name__ == '__main__':
    main()
