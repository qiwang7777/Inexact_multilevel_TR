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


def solve_state(mesh,z,tol=1e-8,max_newton=30,verbose=False,u0=None):
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
        if nr<=tol: return u
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


def solve_adjoint(mesh,u,w_value=-1.0,rtol=1e-8,maxiter=1000,verbose=False):
    #K=assemble_stiffness(mesh)
    K = get_stiffness(mesh)
    _,J=assemble_nonlinear_terms(mesh,u)
    I=mesh.interior
    A=(K+J)[I][:,I]
    b=assemble_p1_load(mesh,u,w_value)[I]
    pI,info=spla.cg(A,b,rtol=rtol,atol=0.0,maxiter=maxiter)
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
                             state_tol=1e-6,adjoint_tol=1e-6,verbose=False):
    u=solve_state(mesh,z,tol=state_tol,verbose=verbose)
    p=solve_adjoint(mesh,u,w_value=w_value,rtol=adjoint_tol,verbose=verbose)
    return u,p,reduced_gradient(mesh,z,p,alpha)


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
    hessian_mode: str = 'diagonal'  # 'diagonal' or 'lbfgs'
    lbfgs_memory: int = 5
    lbfgs_cg_maxiter: int = 50
    lbfgs_cg_rtol: float = 1e-3
    progress_every: int = 25  # Print finest-level convergence diagnostics; 0 disables.
    fine_every: int = 5  # Finest-level relaxation period; 0 disables
    # Experimental adaptive switch from coarse corrections to finest-level L-BFGS.
    # A failure means a coarse-entry eligibility test was actually evaluated
    # and failed. Scheduled fine-only iterations do not count as failures.
    lbfgs_adaptive_restart: bool = True
    lbfgs_restart_min_coarse: int = 5
    lbfgs_restart_failures: int = 2
    lbfgs_switch_to_fine: bool = True


@dataclass
class Evaluation:
    F: float
    g: np.ndarray  # inexact gradient of f
    G: np.ndarray  # inexact gradient of f + exact Moreau gradient
    e: float       # prescribed error budget (not numerically certified)


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


@dataclass
class OuterResult:
    z: np.ndarray
    history: list=field(default_factory=list)
    converged: bool=False


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

    def matvec(self, v):
        return self.R @ self.fine_hessian.matvec(self.P @ v)


class HessianPlusDiagonal:
    """For smoothed problems: L-BFGS model of f plus explicit Moreau curvature."""
    def __init__(self, hessian, diagonal):
        self.hessian = hessian
        self.diagonal = diagonal
        self.bound = hessian.bound + float(np.max(diagonal))

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
        self.restart_events = []
        self.adaptive_checkpoint = None
        # Per-level trial states: objective(z_trial) already solves the PDE.
        # Reuse that state for evaluate(z_trial) if the step was accepted.
        # Only one entry per level is retained; no accumulating memory cost.
        self._trial_state_cache = {}
        self.state_cache_hits = {ell: 0 for ell in range(len(self.levels))}
        self.counters = dict(eligible=0, attempted=0, used=0, fallback=0)
        self.cost = {
            ell:{'eval_calls':0,'eval_time':0.0,
                 'obj_calls':0,'obj_time':0.0}
            for ell in range(len(self.levels))
            }


    def objective(self, ell, z, mu):
        """Trial-point objective; solve state, but not adjoint."""
        t0 = time.perf_counter()
        c=self.cfg
        if not np.all(np.isfinite(z)):
            return np.inf
        else:
            mesh=self.levels[ell].mesh
            u=solve_state(mesh,z,tol=c.state_tol)
            # Save independent snapshots. A rejected trial can never be reused
            # for a different z because evaluate checks exact equality.
            self._trial_state_cache[ell] = (np.asarray(z).copy(), u.copy())
            result = value_f(mesh,z,u,c.alpha,c.desired_state)
            if not c.smooth_only:
                result += moreau_value(mesh,z,mu,c.beta)
        self.cost[ell]['obj_calls']+=1
        self.cost[ell]['obj_time']+= time.perf_counter()-t0
        return result

    def evaluate(self, ell, z, mu, tau):
        """One state+adjoint solve; prescribed error budget is NOT verified."""
        t0 = time.perf_counter()
        c=self.cfg
        mesh=self.levels[ell].mesh
        cached = self._trial_state_cache.pop(ell, None)
        if cached is not None and np.array_equal(cached[0], z):
            # The same state was computed during objective(z_trial).
            # Only the adjoint solve and reduced-gradient assembly remain.
            u = cached[1]
            p = solve_adjoint(mesh, u, w_value=c.desired_state,
                              rtol=c.adjoint_tol)
            g = reduced_gradient(mesh, z, p, c.alpha)
            self.state_cache_hits[ell] += 1
        else:
            u,p,g=compute_inexact_gradient(
                mesh,z,alpha=c.alpha,w_value=c.desired_state,
                state_tol=c.state_tol,adjoint_tol=c.adjoint_tol)
        F=value_f(mesh,z,u,c.alpha,c.desired_state)#+moreau_value(mesh,z,mu,c.beta)
        if c.smooth_only:
            G= g.copy()
        else:
            F += moreau_value(mesh,z,mu,c.beta)
            G = g+moreau_grad(z,mu,c.beta)
        result = Evaluation(F,g,G,c.e_fraction*tau)
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
            
            a = np.abs(z)
            threshold = mu*c.beta
            curved = (a <= threshold) | (a >= 25.0 + threshold)
            return c.curvature + curved.astype(float)/mu

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

    def make_model(self, ell, z, mu):
        """Return diagonal ndarray (baseline) or Hessian operator (L-BFGS)."""
        if self.cfg.hessian_mode == 'diagonal':
            return self.model_diagonal(z, mu)
        if self.cfg.hessian_mode != 'lbfgs':
            raise ValueError('unknown hessian_mode')
        areas, _ = element_geometry(self.levels[ell].mesh)
        H = LBFGSHessian(areas, self.cfg.curvature, self.lbfgs_pairs[ell])
        if self.cfg.smooth_only:
            return H
        a = np.abs(z)
        threshold = mu * self.cfg.beta
        curved = (a <= threshold) | (a >= 25.0 + threshold)
        D = curved.astype(float) / mu  # Hessian f is already in H.
        return HessianPlusDiagonal(H, D)

    @staticmethod
    def hessian_matvec(B, v):
        return B * v if isinstance(B, np.ndarray) else B.matvec(v)

    @staticmethod
    def hessian_bound(B):
        return float(np.max(B)) if isinstance(B, np.ndarray) else B.bound

    def tr_cg_step(self, ell, G, B, delta):
        """Steihaug truncated CG in the weighted L2 geometry.

        Approximate trust-region solution for a positive Hessian operator.
        Returns a feasible Cauchy-or-better step if the operator is SPD.
        """
        inner = lambda u, v: p0_inner(self.levels[ell].mesh, u, v)
        s = np.zeros_like(G)
        r = G.copy()
        d = -r.copy()
        rr = inner(r, r)
        initial_norm = np.sqrt(max(rr, 0.0))
        if initial_norm == 0 or delta <= 0:
            return s

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
            alpha = rr / dBd
            s_next = s + alpha*d
            if self.norm(ell, s_next) >= delta:
                return boundary_step(s, d)
            r_next = r + alpha*Bd
            rr_next = inner(r_next, r_next)
            if np.sqrt(max(rr_next, 0.0)) <= self.cfg.lbfgs_cg_rtol * initial_norm:
                return s_next
            d = -r_next + (rr_next / rr)*d
            s, r, rr = s_next, r_next, rr_next
        return s

    def model_step(self, ell, G, Bdiag, delta):
        """Exact diagonal TR step or matrix-free truncated-CG L-BFGS step."""
        if delta <= 0:
            return np.zeros_like(G)
        if not isinstance(Bdiag, np.ndarray):
            return self.tr_cg_step(ell, G, Bdiag, delta)
        s = -G/Bdiag
        if self.norm(ell,s) <= delta:
            return s
        lo, hi = 0., 1.
        while self.norm(ell,G/(Bdiag+hi)) > delta:
            hi *= 2.
        for _ in range(55):
            mid = .5*(lo+hi)
            if self.norm(ell,G/(Bdiag+mid)) > delta:
                lo = mid
            else:
                hi = mid
        return -G/(Bdiag+hi)

    def pred(self, ell, G, s, Bdiag):
        mesh = self.levels[ell].mesh
        Bs = self.hessian_matvec(Bdiag, s)
        return -p0_inner(mesh,G,s)-.5*p0_inner(mesh,s,Bs)

    def fcd_ok(self, ell, G, s, delta, Bdiag):
        c = self.cfg
        gnorm = self.norm(ell, G)
        rhs = c.kappa_fcd*gnorm*min(gnorm/(1+self.hessian_bound(Bdiag)),delta)
        return self.pred(ell,G,s,Bdiag)+1e-12>=rhs

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
        pending_pair = None  # (accepted step, smooth gradient at old control)
        # These counters are local to this call: on the finest level, one
        # call corresponds to one optimization stage at fixed (mu, tau).
        finest = ell == len(self.levels) - 1
        if finest:
            # Avoid exposing a checkpoint from a previous smoothing stage.
            self.adaptive_checkpoint = None
        accepted_coarse = 0
        failed_eligibility_tests = 0
        did_adaptive_restart = False
        coarse_phase_finished = False
        for k in range(limit):
            if ev_cached is None:
                ev_cached = self.evaluate(ell,z,mu,tau)
            ev = ev_cached
            # The new fine gradient is needed anyway for stopping. Use it now
            # to record the previous accepted step's secant pair at zero extra solves.
            if pending_pair is not None:
                old_step, old_g = pending_pair
                self.add_lbfgs_pair(ell, old_step, ev.g - old_g)
                pending_pair = None
            # The adaptive restart is evaluated below, after computing the
            # actual coarse-entry test. It must not depend on iteration number.
            stationarity = self.norm(ell, ev.G) + ev.e
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
            else:
                Bdiag = self.make_model(ell, z, mu)

            

            #step = self.model_step(ell, ev.G, Bdiag, delta)
            #predicted = self.pred(ell, ev.G, step, Bdiag)
            step = None
            predicted = None
            mode = 'current'
            zeta = np.nan
            zeta_smooth = np.nan
            zeta_moreau = np.nan
            eligible = False
            reject_reason = ''
            force_fine = (finest and c.fine_every > 0 and (k + 1) % c.fine_every == 0)
            # Set only after an adaptive restart, not at a hand-picked k.
            disable_coarse = coarse_phase_finished
            reset_this_iteration = False

            if (ell > 0 and (not c.smooth_only or finest)
                    and not force_fine and not disable_coarse):
                
                R, P = self.levels[ell].R, self.levels[ell].P
                zc = R @ z
                cev = self.evaluate(ell-1, zc, mu, tau)
                zeta = self.norm(ell-1, cev.G - R @ ev.G)
                # Decompose the uncorrected first-order inconsistency:
                # G_c - R G_f = (g_c - R g_f) + (grad phi_mu(z_c) - R grad phi_mu(z_f)).
                # The component NORMS need not add to zeta, because of cancellation.
                if finest:
                    zeta_smooth = self.norm(ell-1, cev.g - R @ ev.g)
                    if not c.smooth_only:
                        dm = (moreau_grad(zc, mu, c.beta)
                              - R @ moreau_grad(z, mu, c.beta))
                        zeta_moreau = self.norm(ell-1, dm)
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

                # Track only eligibility tests that actually happened; periodic
                # fine-level relaxation must not count as a failed test.
                if finest and c.hessian_mode == 'lbfgs':
                    failed_eligibility_tests = (
                        0 if eligible else failed_eligibility_tests + 1
                    )
                    trigger_restart = (
                        c.lbfgs_adaptive_restart
                        and not did_adaptive_restart
                        and accepted_coarse >= c.lbfgs_restart_min_coarse
                        and failed_eligibility_tests >= c.lbfgs_restart_failures
                    )
                    if trigger_restart:
                        # Capture independent copies BEFORE clearing the memory;
                        # useful for the fresh-vs-preserved restart comparison.
                        pairs_before = [
                            (s.copy(), y.copy())
                            for s, y in self.lbfgs_pairs[ell]
                        ]
                        self.adaptive_checkpoint = {
                            'k': k,
                            'mu': float(mu),
                            'tau': float(tau),
                            'z': z.copy(),
                            'delta': float(delta),
                            'pairs': pairs_before,
                            'coarse_accepted': accepted_coarse,
                            'failed_tests': failed_eligibility_tests,
                            'zeta': float(zeta),
                            'fine_gradient_norm': float(gf),
                        }
                        self.lbfgs_pairs[ell].clear()
                        did_adaptive_restart = True
                        reset_this_iteration = True
                        if c.lbfgs_switch_to_fine:
                            coarse_phase_finished = True
                        # Bdiag was assembled before this eligibility check:
                        # rebuild it so THIS step uses the fresh Hessian.
                        Bdiag = self.make_model(ell, z, mu)
                        event = {
                            'k': k, 'mu': float(mu),
                            'coarse_accepted': accepted_coarse,
                            'failed_tests': failed_eligibility_tests,
                            'pairs_cleared': len(pairs_before),
                            'zeta_over_gradient': float(zeta/max(gf, 1e-30)),
                        }
                        self.restart_events.append(event)
                        if c.verbose:
                            print(
                                f"  Adaptive L-BFGS reset: k={k}, "
                                f"coarse_accepted={accepted_coarse}, "
                                f"failed_tests={failed_eligibility_tests}, "
                                f"pairs_cleared={len(pairs_before)}, "
                                f"zeta/||Gf||={event['zeta_over_gradient']:.4f}"
                            )

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
                if eligible and not coarse_phase_finished:
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
                    coarse_pred = self.pred(ell-1, cev.G, sc, Bc)
                    #current_pred = self.pred(ell, ev.G, step, Bdiag)
                    fine_pred = self.pred(ell,ev.G,sf,Bdiag)
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
                    fcd = self.fcd_ok(ell-1, cev.G, sc, inherited, Bc)
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
                step = self.model_step(ell,ev.G,Bdiag,delta)
                predicted = self.pred(ell,ev.G, step, Bdiag)
            if not self.fcd_ok(ell, ev.G, step, delta, Bdiag) and mode == 'current':
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
            F_trial = self.objective(ell, trial, mu)
            cred = ev.F - F_trial
            rho = cred / predicted
            accepted = bool(rho >= c.eta1)
            #delta_before = delta
            
            if accepted:
                # Defer the gradient difference to the next loop iteration.
                # Rejected steps do NOT contribute L-BFGS secant pairs.
                if c.hessian_mode == 'lbfgs':
                    pending_pair = (step.copy(), ev.g.copy())
                if finest and mode == 'coarse':
                    accepted_coarse += 1
                    failed_eligibility_tests = 0
                z = trial
                ev_cached = None
                
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
                                     zeta=zeta,zeta_smooth=zeta_smooth,
                                     zeta_moreau=zeta_moreau,eligible=eligible,
                                     rejection=reject_reason,step_ratio=step_ratio,
                                     coarse_disabled=coarse_phase_finished,
                                     lbfgs_reset=reset_this_iteration))
            if finest and c.verbose and c.progress_every > 0 and (
                k < 3 or (k+1) % c.progress_every == 0
            ):
                print(f'  progress: mu={mu:.5g}, k={k:4d}, '
                      f'F={ev.F:.8e}, stat={stationarity:.3e}, '
                      f'rho={rho:.3e}, delta={delta:.3e}, '
                      f'accepted={accepted}, mode={mode}', flush=True)
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
            return InnerResult(z,limit,np.nan,False, 'coarse iteration limit')
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

    def solve_outer(self,z0,mu0,tau0,eps,
                    theta_mu=.5,theta_tau=.5,r=1.,max_outer=None):
        """Outer Moreau-Yosida algorithm in user's pseudocode."""
        if not (mu0>0 and tau0>0 and eps>0 and r>0
                and 0<theta_mu<1 and 0<theta_tau<1):
            raise ValueError('invalid smoothing, tolerance, or contraction parameters')
        c=self.cfg
        h=len(self.levels)-1
        z=np.asarray(z0,dtype=float).copy()
        mu,tau=mu0,tau0
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
            # The outer error budget can be chosen based on eps, not merely tau.
            test_tol=min(tau,eps)
            stationarity,psi,e=self.original_stationarity(z,r,test_tol)
            history.append(dict(j=j,mu=mu,tau=tau,F_mu=self.objective(h,z,mu),
                                inner_converged=low.converged,
                                inner_reason=low.reason,
                                psi=psi,e=e,stationarity_upper=stationarity,
                                z_start=z_start.copy(),
                                z_end = z.copy(),
                                inner_iterations=low.iterations,
                                inner_final_residual=low.residual_bound,))
            if c.verbose:
                print(f'  outer prox residual={psi:.3e}, '
                      f'prescribed e={e:.3e}, total={stationarity:.3e}')
            if not low.converged:
                if c.verbose:
                    print(f'  Inner IRMTR did not converge: {low.reason}; '
                          'stopping outer smoothing without reducing mu.')
                return OuterResult(z,history,False)
            if stationarity<=eps:
                return OuterResult(z,history,True)
            mu*=theta_mu
            tau*=theta_tau
        return OuterResult(z,history,False)




import time


def objective_breakdown(solver, ell, z, mu):
    """Report smoothed and ORIGINAL nonsmooth objectives separately.

    The original objective is +infinity outside [-25,25]; the Moreau
    envelope is finite there, so they must not be conflated.
    """
    mesh = solver.levels[ell].mesh
    c = solver.cfg
    areas, _ = element_geometry(mesh)
    u = solve_state(mesh, z, tol=c.state_tol)
    smooth = value_f(mesh, z, u, c.alpha, c.desired_state)
    moreau = moreau_value(mesh, z, mu, c.beta)
    violation = float(np.max(np.maximum(np.abs(z) - 25.0, 0.0)))
    original_phi = (float(c.beta * np.dot(areas, np.abs(z)))
                    if violation <= 1e-12 else np.inf)
    original = smooth + original_phi
    prox_distance = solver.norm(ell, z - prox_phi(z, mu, c.beta))
    return dict(smooth=smooth, moreau=moreau,
                smoothed=smooth+moreau, original=original,
                box_violation=violation, prox_distance=prox_distance)



def plot_single_vs_multilevel(results, args):
    r"""Produce ONLY two original-size comparison figures with larger text, PNG + PDF.

    Figure 1: stage-by-stage iterations and original proximal stationarity.
    Figure 2: inner residual histories at selected Moreau parameters.

    All legends are above the axes, never over convergence curves. Original
    physical figure dimensions remain unchanged: outer 13 x 4.8 in,
    inner (two panels) 13.6 x 4.8 in. Captions belong in LaTeX.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D

    if len(results) != 2:
        raise ValueError('Comparison requires exactly two solver results')

    font_pt = float(args.plot_fontsize)
    width_in = float(args.plot_width)
    if font_pt <= 0 or width_in <= 0:
        raise ValueError('--plot-fontsize and --plot-width must be positive')

    # Preserve the original comparison figure aspect and physical height.
    # To achieve ~10 pt text after halving a 13-in PDF in LaTeX, use 20 pt here.
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'font.size': font_pt,
        'axes.labelsize': font_pt,
        'axes.titlesize': font_pt,
        'xtick.labelsize': font_pt - 1,
        'ytick.labelsize': font_pt - 1,
        'legend.fontsize': font_pt - 1.3,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
    })

    pairs = [
        ('Single-level TR', results[0], 'C0'),
        ('Multilevel IRMTR', results[1], 'C1'),
    ]
    plot_dir = args.plot_dir
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    def store(fig, stem):
        # High-resolution PNG for preview and vector PDF for the paper.
        # Keep the *exact* original physical figure dimensions. A top band
        # holds the shared method legend so legends never cover data.
        fig.tight_layout(pad=0.7, w_pad=1.0, rect=(0, 0, 1, 0.88))
        if plot_dir:
            for ext in ('.pdf', '.png'):
                path = os.path.join(plot_dir, stem + ext)
                fig.savefig(path)
                print('Saved comparison plot:', path)

    stages_by_name = {}
    for name, row, color in pairs:
        if args.experiment == 'continuation':
            stages_by_name[name] = row['result'].history
        else:
            r = row['result']
            stages_by_name[name] = [
                dict(j=0, mu=args.mu, tau=args.tau,
                     inner_iterations=r.iterations,
                     inner_converged=r.converged,
                     inner_final_residual=r.residual_bound,
                     stationarity_upper=np.nan)
            ]

    # Save minimal plotting histories so typography can be changed later
    # without repeating thousands of PDE-constrained optimization steps.
    if plot_dir and not args.replot_data:
        import json
        saved = {
            'version': 1, 'experiment': args.experiment, 'eps': args.eps,
            'runs': []
        }
        for name, row, color in pairs:
            stages = []
            for rec in stages_by_name[name]:
                stages.append({
                    'j': int(rec['j']),
                    'mu': float(rec['mu']),
                    'tau': float(rec['tau']),
                    'inner_iterations': int(rec['inner_iterations']),
                    'inner_converged': bool(rec['inner_converged']),
                    'inner_final_residual': float(rec['inner_final_residual']),
                    'stationarity_upper': float(rec['stationarity_upper'])
                })
            h = len(row['solver'].levels) - 1
            trace = [dict(level=h, mu=float(r['mu']), k=int(r['k']),
                          residual=float(r['residual']))
                     for r in row['solver'].history if r['level'] == h]
            saved['runs'].append(dict(name=name, stages=stages,
                                      levels=h+1, trace=trace))
        data_path = os.path.join(plot_dir, 'single_vs_multi_plot_data.json')
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(saved, f, allow_nan=True)
        print('Saved reusable plot histories:', data_path)

    # FIGURE 1: Both solvers, continuation work and stationarity.
    fig1, (ax_work, ax_prox) = plt.subplots(
        1, 2, figsize=(width_in, 4.8))
    seen_stages = {}
    for name, row, color in pairs:
        stages = stages_by_name[name]
        js = np.array([r['j'] for r in stages], dtype=int)
        work = np.array([r['inner_iterations'] for r in stages], dtype=float)
        outer = np.array([r['stationarity_upper'] for r in stages], dtype=float)
        for rec in stages:
            seen_stages[int(rec['j'])] = float(rec['mu'])
        short_name = 'Single TR' if name.startswith('Single') else 'Multi IRMTR'
        ax_work.plot(js, work, '-o', color=color, label=short_name)
        if np.any(np.isfinite(outer)):
            ax_prox.semilogy(js, np.maximum(outer, 1e-16), '-o',
                            color=color, label=short_name)
        for rec in stages:
            if not rec.get('inner_converged', True):
                ax_work.scatter([rec['j']], [rec['inner_iterations']],
                                color=color, marker='x', s=52, zorder=5)
                if np.isfinite(rec['stationarity_upper']):
                    ax_prox.scatter([rec['j']], [rec['stationarity_upper']],
                                    color=color, marker='x', s=52, zorder=5)

    positions = sorted(seen_stages)
    tick_labels = [f'{seen_stages[j]:.4g}' for j in positions]
    for ax in (ax_work, ax_prox):
        ax.set_xticks(positions, tick_labels)
        ax.set_xlabel(r'Smoothing parameter $\mu_j$')
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.margins(x=0.11)
    ax_work.set_title('Inner iterations')
    ax_work.set_ylabel('Finest-level TR iterations')
    ax_work.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    if args.experiment == 'continuation':
        ax_prox.axhline(args.eps, linestyle='--', linewidth=1.25,
                        color='0.3', label=rf'$\varepsilon={args.eps:g}$')
        ax_prox.set_title('Original proximal stationarity')
        ax_prox.set_ylabel(r'$\psi_r(z_j)+e_j$')
    else:
        ax_prox.text(0.5, 0.5, 'Available in continuation mode',
                     ha='center', va='center', transform=ax_prox.transAxes)
        ax_prox.set_title('Outer convergence')
        ax_prox.set_xticks([])
    legend_handles = [
        Line2D([0], [0], color='C0', marker='o', label='Single-level TR'),
        Line2D([0], [0], color='C1', marker='o', label='Multilevel IRMTR'),
    ]
    if args.experiment == 'continuation':
        legend_handles.append(Line2D([0], [0], color='0.3',
                                     linestyle='--', linewidth=1.25,
                                     label=rf'$\varepsilon={args.eps:g}$'))
    fig1.legend(handles=legend_handles, loc='upper center',
                bbox_to_anchor=(0.5, 0.985), ncol=len(legend_handles),
                frameon=False, columnspacing=2, handlelength=1.5)
    store(fig1, 'single_vs_multi_outer')

    # FIGURE 2: Inner stationarity at the same Moreau stages.
    available_mu = [[rec['mu'] for rec in stages_by_name[name]]
                    for name, row, color in pairs]
    common_mus = [mu for mu in available_mu[0]
                  if any(np.isclose(mu, other, rtol=1e-12, atol=0)
                         for other in available_mu[1])]
    selected_mus = (common_mus[-2:] if len(common_mus) >= 2 else common_mus)
    if args.compare_mus:
        selected_mus = args.compare_mus
    if not selected_mus:
        print('No common Moreau stages: inner comparison figure unavailable.')
        plt.show()
        return

    n_panels = len(selected_mus)
    fig2, axes = plt.subplots(1, n_panels,
                             figsize=(width_in * (6.8 / 6.5) if n_panels > 1 else 6.8, 4.8),
                             squeeze=False)
    for ax, target_mu in zip(axes[0], selected_mus):
        last_tau = None
        found_any = False
        for name, row, color in pairs:
            short_name = 'Single TR' if name.startswith('Single') else 'Multi IRMTR'
            stages = stages_by_name[name]
            match = next((s for s in stages
                          if np.isclose(float(s['mu']), float(target_mu),
                                        rtol=1e-12, atol=0)), None)
            if match is None:
                ax.plot([], [], color=color, label=f'{short_name} (not reached)')
                continue
            level = len(row['solver'].levels) - 1
            trace = [r for r in row['solver'].history
                     if r['level'] == level and
                     np.isclose(r['mu'], target_mu, rtol=1e-12, atol=0)]
            if not trace:
                ax.plot([], [], color=color, label=f'{short_name} (no trace)')
                continue
            xx = [r['k'] for r in trace]
            yy = [r['residual'] for r in trace]
            stop_iter = int(match['inner_iterations'])
            stop_stat = match.get('inner_final_residual', np.nan)
            if np.isfinite(stop_stat) and (not xx or stop_iter > xx[-1]):
                xx.append(stop_iter)
                yy.append(float(stop_stat))
            ax.semilogy(xx, np.maximum(yy, 1e-16), color=color,
                        linewidth=1.25, label=short_name)
            last_tau = float(match['tau'])
            found_any = True
        if last_tau is not None:
            ax.axhline(last_tau, color='0.3', linestyle='--', linewidth=1.2)
            # Tolerance label next to its own dashed line; it is not in a
            # floating legend, so it cannot obscure the stationarity curves.
            ax.annotate(rf'$\tau_j={last_tau:g}$',
                        xy=(0.03, last_tau), xycoords=('axes fraction', 'data'),
                        xytext=(0, 5), textcoords='offset points',
                        ha='left', va='bottom', fontsize=font_pt-3,
                        color='0.25')
        stage_title = rf'$\mu={target_mu:.5g}$'
        ax.set_title(stage_title, fontsize=font_pt-1.5)
        ax.set_xlabel('Finest-level TR iteration')
        ax.set_ylabel(r'$\|G_{\mu}(z_k)\|+e_j$')
        ax.grid(True, alpha=0.25, linewidth=0.6)
        if not found_any:
            ax.text(0.5, 0.5, 'Stage not reached',
                    transform=ax.transAxes, ha='center', va='center')
    # One figure-level legend, outside both axes, preserves readable fonts
    # without covering either the blue or orange convergence history.
    fig2.legend(handles=[
        Line2D([0], [0], color='C0', linewidth=1.5, label='Single-level TR'),
        Line2D([0], [0], color='C1', linewidth=1.5, label='Multilevel IRMTR'),
    ], loc='upper center', bbox_to_anchor=(0.5, 0.985), ncol=2,
       frameon=False, columnspacing=3, handlelength=1.8)
    store(fig2, 'single_vs_multi_inner')
    plt.show()


def main():
    """Compare L-BFGS TR and adaptive IRMTR on the FULL nonsmooth objective.

    The default is Moreau continuation with BOTH solvers, followed by two
    comparison-only figures (outer stages and inner stationarity).
    """
    parser = argparse.ArgumentParser(
        description='Full L1 + box Moreau problem: single TR versus adaptive multilevel IRMTR'
    )
    parser.add_argument('--solver', choices=('both', 'single', 'multi'), default='both',
                        help='Run single-level, multilevel, or both solvers')
    parser.add_argument('--progress-every', type=int, default=25,
                        help='Print a finest-level progress row every N iterations; 0 disables')
    parser.add_argument('--fine-every', type=int, default=5,
                        help='Schedule a fine-level step every N iterations also for full objective; 0 disables')
    parser.add_argument('--experiment', choices=('fixed', 'continuation'), default='continuation',
                        help='fixed: one Moreau-smoothed objective; continuation: original nonsmooth target')
    parser.add_argument('--base-n', type=int, default=8)
    parser.add_argument('--full', action='store_true',
                        help='Use very large n=(64,128,256,512), as in the original script')
    parser.add_argument('--max-inner', type=int, default=4000)
    parser.add_argument('--max-coarse-inner', type=int, default=1)
    parser.add_argument('--max-outer', type=int, default=8)
    parser.add_argument('--curvature', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--kappa-ml', type=float, default=.25)
    parser.add_argument('--kappa-et', type=float, default=.2)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--hessian-mode', choices=('diagonal', 'lbfgs'), default='lbfgs')
    parser.add_argument('--lbfgs-memory', type=int, default=5)
    parser.add_argument('--lbfgs-cg-maxiter', type=int, default=50)
    parser.add_argument('--lbfgs-cg-rtol', type=float, default=1e-3)
    parser.add_argument('--lbfgs-restart-min-coarse', type=int, default=5)
    parser.add_argument('--lbfgs-restart-failures', type=int, default=2)
    parser.add_argument('--no-adaptive-restart', action='store_true')
    parser.add_argument('--no-switch-to-fine', action='store_true')
    # Parameters for the controlled fixed-Moreau experiment.
    parser.add_argument('--mu', type=float, default=0.0125)
    parser.add_argument('--tau', type=float, default=1e-5)
    # Parameters for solving the original nonsmooth problem via continuation.
    parser.add_argument('--mu0', type=float, default=0.1)
    parser.add_argument('--tau0', type=float, default=8e-3)
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--theta-mu', type=float, default=.5)
    parser.add_argument('--theta-tau', type=float, default=.5)
    parser.add_argument('--prox-r', type=float, default=1.)
    parser.add_argument('--restart-probes', action='store_true',
                        help='In fixed mode, run additional fresh/preserved-memory checkpoint solves')
    parser.add_argument('--plots', action='store_true', default=True,
                        help='Show comparison-only plots (enabled by default)')
    parser.add_argument('--no-plots', action='store_false', dest='plots',
                        help='Disable comparison plots')
    parser.add_argument('--plot-dir', type=str, default=None,
                        help='Directory to save two comparison PNGs and vector PDFs')
    parser.add_argument('--plot-fontsize', type=float, default=20.0,
                        help='Plot text size in pt; default 20 gives ~10 pt when 13-in PDF is scaled to 6.5 in')
    parser.add_argument('--plot-width', type=float, default=13.0,
                        help='Original outer figure width in inches (default: 13; normally leave unchanged)')
    parser.add_argument('--replot-data', type=str, default=None,
                        help='Replot a saved single_vs_multi_plot_data.json without rerunning either solver')
    parser.add_argument('--compare-mus', nargs='*', type=float, default=None,
                        help='Moreau values for inner comparison, e.g. --compare-mus 0.025 0.0125. '
                             'Default: two latest common stages')
    args, _unknown = parser.parse_known_args()  # Spyder / runfile compatible

    if args.mu <= 0 or args.tau <= 0 or args.beta < 0:
        parser.error('--mu, --tau must be > 0 and --beta must be >= 0')
    if args.lbfgs_restart_failures < 1 or args.lbfgs_restart_min_coarse < 0:
        parser.error('adaptive restart counts must be nonnegative (failures >= 1)')
    if args.fine_every < 0:
        parser.error('--fine-every must be >= 0')

    # Replot saved histories using new font sizes, without any PDE solves.
    if args.replot_data:
        import json
        from types import SimpleNamespace
        with open(args.replot_data, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        if saved.get('version') != 1 or len(saved.get('runs', [])) != 2:
            parser.error('--replot-data is not a valid two-solver comparison history')
        args.experiment = saved['experiment']
        args.eps = float(saved['eps'])
        rows = []
        for run in saved['runs']:
            stages = run['stages']
            if args.experiment == 'continuation':
                result = SimpleNamespace(history=stages)
            else:
                st = stages[0]
                result = SimpleNamespace(
                    iterations=st['inner_iterations'],
                    converged=st['inner_converged'],
                    residual_bound=st['inner_final_residual'])
            solver = SimpleNamespace(levels=[None]*run['levels'],
                                     history=run['trace'])
            rows.append(dict(result=result, solver=solver))
        plot_single_vs_multilevel(rows, args)
        return

    ns = ((64,128,256,512) if args.full else
          (args.base_n, 2*args.base_n, 4*args.base_n, 8*args.base_n))
    meshes = [create_unit_square_mesh(n) for n in ns]
    finest = meshes[-1]
    z0 = 2.0*np.ones(len(finest.elements))

    def make_config():
        return Config(
            beta=args.beta,
            desired_state=-1.,
            max_inner=args.max_inner,
            max_coarse_inner=args.max_coarse_inner,
            max_outer=args.max_outer,
            curvature=args.curvature,
            smooth_only=False,  # IMPORTANT: include Moreau envelope in F and G
            hessian_mode=args.hessian_mode,
            lbfgs_memory=args.lbfgs_memory,
            lbfgs_cg_maxiter=args.lbfgs_cg_maxiter,
            lbfgs_cg_rtol=args.lbfgs_cg_rtol,
            progress_every=args.progress_every,
            fine_every=args.fine_every,
            lbfgs_adaptive_restart=not args.no_adaptive_restart,
            lbfgs_restart_min_coarse=args.lbfgs_restart_min_coarse,
            lbfgs_restart_failures=args.lbfgs_restart_failures,
            lbfgs_switch_to_fine=not args.no_switch_to_fine,
            kappa_ml=args.kappa_ml,
            kappa_et=args.kappa_et,
            verbose=not args.quiet
        )

    print('Mesh n:', ns)
    print('Control DOFs:', [len(m.elements) for m in meshes])
    print('\n========== FULL OBJECTIVE TEST ==========')
    print('Objective: f(z) + phi_mu(z),  phi(z) = beta ||z||_L1 + I_[-25,25](z)')
    print(f'Experiment: {args.experiment}  beta={args.beta:.6g}  '
          f'Hessian mode={args.hessian_mode}  memory={args.lbfgs_memory}')
    print('L-BFGS model: smooth reduced Hessian approximation + exact Moreau diagonal')
    print('Scheduled fine-level relaxation: every', args.fine_every, 'iterations (0 disables)')
    print('State caching: reuse accepted trial PDE states')
    print('Adaptive restart:', not args.no_adaptive_restart,
          'min accepted coarse:', args.lbfgs_restart_min_coarse,
          'consecutive failed eligibility:', args.lbfgs_restart_failures,
          'switch to fine:', not args.no_switch_to_fine)
    if args.experiment == 'fixed':
        print(f'Fixed Moreau stage: mu={args.mu:.6g}, tau={args.tau:.6g}')
        print('NOTE: fixed mu minimizes the MOREAU-SMOOTHED objective, not exactly f + phi.')
    else:
        print(f'Continuation: mu0={args.mu0:.6g}, tau0={args.tau0:.6g}, '
              f'eps={args.eps:.6g}, max_outer={args.max_outer}')

    single = IRMTRMoreau([finest], make_config())
    multi = IRMTRMoreau(meshes, make_config())
    results = []
    candidates = [('Single TR', single), ('Multilevel IRMTR', multi)]
    for label, solver in candidates:
        if args.solver == 'single' and label != 'Single TR':
            continue
        if args.solver == 'multi' and label != 'Multilevel IRMTR':
            continue
        h = len(solver.levels) - 1
        t0 = time.perf_counter()
        if args.experiment == 'fixed':
            result = solver.irmtr(h, z0.copy(), mu=args.mu, tau=args.tau)
            elapsed = time.perf_counter() - t0
            z, mu_last = result.z, args.mu
            inner_count = result.iterations
            converged = result.converged
            why = result.reason
            stage_history = None
        else:
            result = solver.solve_outer(
                z0.copy(), mu0=args.mu0, tau0=args.tau0, eps=args.eps,
                theta_mu=args.theta_mu, theta_tau=args.theta_tau,
                r=args.prox_r, max_outer=args.max_outer
            )
            elapsed = time.perf_counter() - t0
            z = result.z
            stage_history = result.history
            mu_last = stage_history[-1]['mu'] if stage_history else args.mu0
            inner_count = sum(s['inner_iterations'] for s in stage_history)
            converged = result.converged
            why = ('original proximal stationarity reached' if converged else
                   (stage_history[-1]['inner_reason'] if stage_history else 'no stages'))
        # Diagnostics below are deliberately outside the timed solve.
        parts = objective_breakdown(solver, h, z, mu_last)
        tau_diag = args.tau if args.experiment == 'fixed' else min(args.tau0, args.eps)
        ev = solver.evaluate(h, z, mu_last, tau_diag)
        psi_total, psi, err = solver.original_stationarity(z, args.prox_r, tau_diag)
        fine_rows = [r for r in solver.history if r['level'] == h]
        coarse_rows = [r for r in fine_rows if r['mode'] == 'coarse']
        rejected = sum(not r['accepted'] for r in fine_rows)
        print(f'\n========== {label} ==========')
        print(f'Converged: {converged}  Reason: {why}')
        print(f'Timed solve (s): {elapsed:.3f}')
        print(f'Finest-level iterations: {inner_count}  coarse steps: {len(coarse_rows)}'
              f'  rejected finest-level steps: {rejected}')
        if args.experiment == 'continuation':
            print('Outer stages:', len(stage_history))
            for s in stage_history:
                print(f"  j={s['j']:2d} mu={s['mu']:.3e} tau={s['tau']:.3e} "
                      f"iters={s['inner_iterations']:4d} "
                      f"inner={s['inner_reason']} psi+e={s['stationarity_upper']:.3e}")
        print(f'  f(z) (smooth component):              {parts["smooth"]:.12e}')
        print(f'  phi_mu(z) (Moreau envelope):          {parts["moreau"]:.12e}')
        print(f'  F_mu(z) = f(z) + phi_mu(z):           {parts["smoothed"]:.12e}')
        print(f'  Original F(z) = f(z) + phi(z):        {parts["original"]:.12e}')
        print(f'  Box-constraint violation max:        {parts["box_violation"]:.3e}')
        print(f'  ||z - prox_(mu phi)(z)||_L2:        {parts["prox_distance"]:.3e}')
        print(f'  Smoothed gradient ||G_mu||_L2:        {solver.norm(h, ev.G):.6e}')
        print(f'  Original prox stationarity psi:       {psi:.6e}')
        print(f'  Original psi + prescribed error:      {psi_total:.6e}')
        print('  Coarse counters:', solver.counters)
        print('  Adaptive restart events:', solver.restart_events)
        print('  Reused trial-state PDE solves per level:', solver.state_cache_hits)
        if label == 'Multilevel IRMTR':
            ks = [(r['mu'],r['k']) for r in coarse_rows]
            print('  (mu, k) of coarse steps:', ks[:60],
                  '...' if len(ks) > 60 else '')
            if args.experiment == 'fixed':
                diagnostics = [r for r in fine_rows if np.isfinite(r['zeta'])]
                shown = {r['k']:r for r in diagnostics[:4] + diagnostics[-4:]}
                for r in shown.values():
                    gf = r['residual'] - solver.cfg.e_fraction*args.tau
                    print(f"    k={r['k']:4d} mode={r['mode']:7s} "
                          f"eligible={r['eligible']} "
                          f"zeta/||G||={r['zeta']/max(gf,1e-30):.4g} "
                          f"zeta_smooth={r['zeta_smooth']:.3e} "
                          f"zeta_moreau={r['zeta_moreau']:.3e}")
        results.append(dict(name=label, solver=solver, result=result,
                            z=z, elapsed=elapsed, iters=inner_count,
                            coarse=len(coarse_rows), converged=converged,
                            original_psi=psi, Fmu=parts['smoothed']))

    print('\n========== FAIR COMPARISON ==========')
    for row in results:
        print(f"{row['name']:17s}: iters={row['iters']:4d}, "
              f"time={row['elapsed']:.3f}s, coarse={row['coarse']:4d}, "
              f"F_mu={row['Fmu']:.10g}, psi={row['original_psi']:.3e}, "
              f"converged={row['converged']}")
    if args.experiment == 'fixed':
        print('Convergence above means ||grad F_mu||+e <= tau, NOT the original prox criterion.')
    else:
        print('Convergence above means original prox stationarity <= eps after a converged inner stage.')

    if (args.restart_probes and args.solver != 'single'
            and args.experiment == 'fixed' and args.hessian_mode == 'lbfgs'):
        from copy import deepcopy
        checkpoint = multi.adaptive_checkpoint
        print('\n========== ADAPTIVE CHECKPOINT RESTARTS ==========')
        if checkpoint is None:
            print('No adaptive reset: no checkpoint available.')
        else:
            print('Reset k:', checkpoint['k'], 'saved curvature pairs:',len(checkpoint['pairs']))
            for preserved in [False, True]:
                cfg = deepcopy(multi.cfg)
                cfg.delta_local = checkpoint['delta']
                probe = IRMTRMoreau([finest], cfg)
                if preserved:
                    probe.lbfgs_pairs[0] = [(s.copy(), y.copy()) for s,y in checkpoint['pairs']]
                initial_pair_count = len(probe.lbfgs_pairs[0])
                resumed = probe.irmtr(0, checkpoint['z'].copy(),
                                       mu=checkpoint['mu'],tau=checkpoint['tau'])
                print(('Preserved' if preserved else 'Fresh') +
                      f' L-BFGS: initial pairs={initial_pair_count}, '
                      f'additional iterations={resumed.iterations}, '
                      f'converged={resumed.converged}, stat={resumed.residual_bound:.3e}')
    if args.plots:
        if len(results) != 2:
            print('Comparison plots require --solver both; no plots generated.')
        else:
            plot_single_vs_multilevel(results, args)


if __name__ == '__main__':
    main()
