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
    kappa_ml: float=.25
    kappa_et: float=.1
    max_outer: int=10
    verbose: bool=True


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


class IRMTRMoreau:
    def __init__(self, meshes, config=None):
        self.cfg = config or Config()
        self.levels = [Level(mesh) for mesh in meshes]
        for ell in range(1, len(self.levels)):
            R, P = p0_transfers(meshes[ell-1], meshes[ell])
            self.levels[ell].R, self.levels[ell].P = R, P
        self.history = []
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
        mesh=self.levels[ell].mesh
        u=solve_state(mesh,z,tol=c.state_tol)
        result = value_f(mesh,z,u,c.alpha,c.desired_state)+moreau_value(mesh,z,mu,c.beta)
        self.cost[ell]['eval_calls']+=1
        self.cost[ell]['eval_time']+= time.perf_counter()-t0
        return result

    def evaluate(self, ell, z, mu, tau):
        """One state+adjoint solve; prescribed error budget is NOT verified."""
        t0 = time.perf_counter()
        c=self.cfg
        mesh=self.levels[ell].mesh
        u,p,g=compute_inexact_gradient(
            mesh,z,alpha=c.alpha,w_value=c.desired_state,
            state_tol=c.state_tol,adjoint_tol=c.adjoint_tol)
        F=value_f(mesh,z,u,c.alpha,c.desired_state)+moreau_value(mesh,z,mu,c.beta)
        result = Evaluation(F,g,g+moreau_grad(z,mu,c.beta),c.e_fraction*tau)
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
        a = np.abs(z)
        threshold = mu*c.beta
        curved = (a <= threshold) | (a >= 25.0 + threshold)
        return c.curvature + curved.astype(float)/mu

    def model_step(self, ell, G, Bdiag, delta):
        """Solve diagonal positive-definite L2 trust-region model exactly.

        min_s <G,s> + 0.5 <s,Bs>, ||s||_L2 <= delta.
        With multiplier lam>=0: s_i=-G_i/(B_i+lam).
        Bisection determines lam when the unconstrained step is outside TR.
        """
        if delta <= 0:
            return np.zeros_like(G)
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
        mesh=self.levels[ell].mesh
        return -p0_inner(mesh,G,s)-.5*p0_inner(mesh,s,Bdiag*s)

    def fcd_ok(self,ell,G,s,delta,Bdiag):
        c=self.cfg
        gnorm=self.norm(ell,G)
        rhs=c.kappa_fcd*gnorm*min(gnorm/(1+np.max(Bdiag)),delta)
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
        for k in range(limit):
            if ev_cached is None:
                ev_cached = self.evaluate(ell,z,mu,tau)
            ev = ev_cached
            #ev = self.evaluate(ell, z, mu, tau)
            if k==0 and ev_initial is not None:
                ev = ev_initial
            else:
                ev = self.evaluate(ell, z, mu, tau)
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
                Bdiag = B_initial.copy()
            else:
                Bdiag = self.model_diagonal(z, mu)

            

            #step = self.model_step(ell, ev.G, Bdiag, delta)
            #predicted = self.pred(ell, ev.G, step, Bdiag)
            step = None
            predicted = None
            mode = 'current'
            zeta = np.nan
            eligible = False
            reject_reason = ''

            if ell > 0: #and mu>=0.05:
                
                R, P = self.levels[ell].R, self.levels[ell].P
                zc = R @ z
                cev = self.evaluate(ell-1, zc, mu, tau)
                zeta = self.norm(ell-1, cev.G - R @ ev.G)
                coarsening_error = self.norm(ell,z-P@zc)
                
                if ell == len(self.levels)-1 and mu <= 0.025:
                    if k < 3 or k % 50 == 0:
                        print(f"mu={mu:.5f},k={k}," 
                              f"||z-PRz|| = {coarsening_error:.6e},"
                              f"bound = {coarsening_error/mu:.6e}")

                        d_smooth = cev.g - R @ ev.g

                        d_moreau = (moreau_grad(zc, mu, c.beta)- R @ moreau_grad(z, mu, c.beta))

                        print(f"mu={mu:.5f}, k={k}, "f"zeta={zeta:.3e}, "f"smooth_gap={self.norm(ell-1, d_smooth):.3e}, "f"moreau_gap={self.norm(ell-1, d_moreau):.3e}")
                gf = self.norm(ell,ev.G)
                gc = self.norm(ell-1,cev.G)
                ml_threshold = c.kappa_ml*min(delta,gf)
                et_threshold = c.kappa_et*gf
                ml_ok = zeta <= ml_threshold
                et_ok = gc >= et_threshold
                eligible = ml_ok and et_ok

                #eligible = (zeta <= c.kappa_ml * min(delta, self.norm(ell, ev.G))
                #            and self.norm(ell-1, cev.G) >= c.kappa_et*self.norm(ell, ev.G))
                if ell== len(self.levels)-1 and mu<= 0.025:
                    if k < 3 or k % 50 == 0:
                        print(
                            f"mu={mu:.5f}, k={k}, "
                            f"zeta={zeta:.3e}, "
                            f"ML threshold={ml_threshold:.3e}, "
                            f"ML condition={ml_ok}, "
                            f"coarse gradient={gc:.3e}, "
                            f"ET threshold={et_threshold:.3e}, "
                            f"ET condition={et_ok}"
                            )
                if eligible:
                    self.counters['eligible'] += 1
                    # For nested uniform P0 elements, ||P||_{L2->L2}=1.
                    inherited = delta
                    self.counters['attempted'] += 1
                    Bc = R @ Bdiag
                    low = self.irmtr(ell-1, zc, mu, tau, inherited_delta=inherited, B_initial=Bc, ev_initial = cev)
                    sc = low.z - zc
                    sf = P @ sc
                    
                    #Bc = self.model_diagonal(zc, mu)
                    #Bc = R @ Bdiag
                    coarse_pred = self.pred(ell-1, cev.G, sc, Bc)
                    #current_pred = self.pred(ell, ev.G, step, Bdiag)
                    #fine_pred = self.pred(ell,ev.G,sf,Bdiag)
                    #discrepancy = abs(fine_pred - coarse_pred)
                    #relative_discrepancy = discrepancy/max(abs(coarse_pred),1e-14)
                    
                   # print(f"level={ell}," f"zeta={zeta:.3e}, "f"step_norm={self.norm(ell-1, sc):.3e}," 
                   #       f"coarse_pred={coarse_pred:.3e},"
                   #       f"fine_pred={fine_pred:.3e},"
                   #       f"relative model discrepancy = {relative_discrepancy:.3e}")
                    feasible = self.norm(ell, sf) <= delta*(1+1e-10)
                    fcd = self.fcd_ok(ell-1, cev.G, sc, inherited, Bc)
                    if feasible and coarse_pred > 1e-14 and fcd:
                        step, predicted, mode = sf, coarse_pred, 'coarse'
                        self.counters['used'] += 1
                    else:
                        self.counters['fallback'] += 1
                        reject_reason = ('infeasible' if not feasible else
                                         'nonpositive coarse prediction' if coarse_pred <= 1e-14 else
                                         'coarse FCD failed')
            if mode == 'current':
                step = self.model_step(ell,ev.G,Bdiag,delta)
                predicted = self.pred(ell,ev.G, step, Bdiag)
            if not self.fcd_ok(ell, ev.G, step, delta, Bdiag) and mode == 'current':
                return InnerResult(z, k, stationarity, False, 'FCD failed')
            if predicted <= 0 or not np.isfinite(predicted):
                return InnerResult(z, k, stationarity, False, 'nonpositive predicted reduction')
            step_ratio = self.norm(ell,step)/max(delta,1e-14)
            trial = z + step
            F_trial = self.objective(ell, trial, mu)
            cred = ev.F - F_trial
            rho = cred / predicted
            accepted = bool(rho >= c.eta1)
            #delta_before = delta
            
            if accepted:
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
                                     accepted=accepted,zeta=zeta,eligible=eligible,
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
        ev = ev_cached if ev_cached is not None else self.evaluate(ell, z, mu, tau)
        final_stat = self.norm(ell, ev.G) + ev.e
        if c.verbose and ell == len(self.levels)-1:
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
            low=self.irmtr(h,z,mu,tau,inherited_delta=np.inf)
            z=low.z
            # The outer error budget can be chosen based on eps, not merely tau.
            test_tol=min(tau,eps)
            stationarity,psi,e=self.original_stationarity(z,r,test_tol)
            history.append(dict(j=j,mu=mu,tau=tau,F_mu=self.objective(h,z,mu),
                                inner_converged=low.converged,
                                inner_reason=low.reason,
                                psi=psi,e=e,stationarity_upper=stationarity))
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

#def main():
#    parser = argparse.ArgumentParser(description='Three-level uncorrected IRMTR for semilinear control')
#    parser.add_argument('--full',action='store_true',help='Use n=64,128,256')
#    parser.add_argument('--base-n',type=int,default=4,help='Coarsest n; levels n,2n,4n')
#    parser.add_argument('--max-inner',type=int,default=4000,help='Finest-level iteration limit per smoothing stage')
#    parser.add_argument('--max-coarse-inner',type=int,default=8,help='Recursive subproblem iteration limit')
#    parser.add_argument('--max-outer',type=int,default=8)
#    parser.add_argument('--curvature',type=float,default=.1)
#    parser.add_argument('--kappa-ml',type=float,default=.25)
#    parser.add_argument('--kappa-et',type=float,default=.1)
#    parser.add_argument('--quiet',action='store_true')
#    parser.add_argument('--no-plots',action='store_true')
#    args,_unknown=parser.parse_known_args()  # works with Spyder IPython runfile
#    ns=(64,128,256) if args.full else (args.base_n,2*args.base_n,4*args.base_n)
#    meshes=[create_unit_square_mesh(n) for n in ns]
#    finest = meshes[-1]
#    print('Mesh n:',ns,'Control DOFs:',[len(m.elements) for m in meshes])
#    cfg=Config(max_inner=args.max_inner,max_coarse_inner=args.max_coarse_inner,
#               max_outer=args.max_outer,curvature=args.curvature,
#               kappa_ml=args.kappa_ml,kappa_et=args.kappa_et,
#               verbose=not args.quiet)
    
    
#    z0=np.zeros(len(meshes[-1].elements))
    #print('Initial smoothed F:',solver.objective(2,z0,.1))
    #print('\n========== SINGLE-LEVEL TR ==========')
    #solver_single = IRMTRMoreau(finest,cfg)
    #start_single = time.perf_counter
#    out_single= solver_single(z0,mu0=.1,tau0=1e-2,eps=1e-3,
#                           theta_mu=.5,theta_tau=.5,r=1.)
#    elapsed_single = time.perf_counter()-start_single
#    print('\n========== MULTILEVEL IRMTR ==========')
#    solver=IRMTRMoreau(meshes,cfg)
#    start = time.perf_counter()
#    out=solver.solve_outer(z0,mu0=.1,tau0=1e-2,eps=1e-3,
#                           theta_mu=.5,theta_tau=.5,r=1.)
#    elapsed = time.perf_counter()-start
    
    
#    print('Outer converged:',out.converged)
#    for r in out.history:print('Outer:',r)
#    print('Coarse recursion counters:',solver.counters)
#    from collections import Counter
#    print('Steps by level and mode:',dict(Counter((r['level'],r['mode']) for r in solver.history)))
#    if not args.no_plots:plot_results(meshes[-1],solver,out)

def main():
    import time
    from collections import Counter
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description='Compare single-level TR and multilevel IRMTR'
    )
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--base-n', type=int, default=16)
    parser.add_argument('--max-inner', type=int, default=4000)
    parser.add_argument('--max-coarse-inner', type=int, default=1)
    parser.add_argument('--max-outer', type=int, default=8)
    parser.add_argument('--curvature', type=float, default=.1)
    parser.add_argument('--kappa-ml', type=float, default=.25)
    parser.add_argument('--kappa-et', type=float, default=.1)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--no-plots', action='store_true')

    args, _unknown = parser.parse_known_args()

    ns = (64, 128, 256) if args.full else (
        args.base_n, 2*args.base_n, 4*args.base_n
    )

    meshes = [create_unit_square_mesh(n) for n in ns]
    finest = meshes[-1]

    print('Mesh n:', ns)
    print('Control DOFs:', [len(m.elements) for m in meshes])

    z0 = 2.0*np.ones(len(finest.elements))

    # Identical parameters for both methods
    def make_config():
        return Config(
            desired_state = -1.0,
            max_inner=args.max_inner,
            max_coarse_inner=args.max_coarse_inner,
            max_outer=args.max_outer,
            curvature=args.curvature,
            kappa_ml=args.kappa_ml,
            kappa_et=args.kappa_et,
            verbose=not args.quiet
        )

    outer_args = dict(
        mu0=.1,
        tau0=8e-3,
        eps=1e-3,
        theta_mu=.5,
        theta_tau=.5,
        r=1.
    )

    # ==========================================
    # 1. Single-level TR
    # ==========================================
    print('\n========== SINGLE-LEVEL TR ==========')

    single = IRMTRMoreau([finest], make_config())

    start = time.perf_counter()
    out_single = single.solve_outer(z0.copy(), **outer_args)
    time_single = time.perf_counter() - start

    # ==========================================
    # 2. Multilevel IRMTR
    # ==========================================
    print('\n========== MULTILEVEL IRMTR ==========')

    multi = IRMTRMoreau(meshes, make_config())

    start = time.perf_counter()
    out_multi = multi.solve_outer(z0.copy(), **outer_args)
    time_multi = time.perf_counter() - start
    
    print("\n========== EXACT SOLUTION TEST ==========")

    for name, solver, out in [("Single TR", single, out_single),("Multilevel IRMTR", multi, out_multi)]:
        h = len(solver.levels) - 1

        error_L2 = solver.norm(h, out.z)
        error_max = np.max(np.abs(out.z))

        print(f"\n{name}")
        print(f"Converged: {out.converged}")
        print(f"L2 control error: {error_L2:.8e}")
        print(f"Maximum control error: {error_max:.8e}")
        print(f"Final smoothed objective: {out.history[-1]['F_mu']:.8e}")
        print(f"Final stationarity: {out.history[-1]['stationarity_upper']:.8e}")
    
    print("\n========== FINEST-LEVEL STEP ANALYSIS ==========")

    fine = len(multi.levels) - 1

    for mode in ('current', 'coarse'):
        rows = [
            h for h in multi.history
            if h['level'] == fine and h['mode'] == mode
            and np.isclose(h['mu'],0.0125)]

        accepted = sum(h['accepted'] for h in rows)

        print(f"\n{mode.upper()} steps")
        print("Trials:", len(rows))
        print("Accepted:", accepted)
        print("Rejected:", len(rows) - accepted)

        if rows:
            print("Acceptance rate:", accepted / len(rows))
            print("Mean rho:", np.mean([h['rho'] for h in rows]))

    # ==========================================
    # 3. Performance comparison
    # ==========================================
    def summarize(solver, out, elapsed):
        h = len(solver.levels) - 1
        counts = Counter(
            item['level'] for item in solver.history
        )
        last = out.history[-1] if out.history else {}

        return {
            'Converged': out.converged,
            'Wall time (s)': round(elapsed, 2),
            'Finest TR iterations': counts.get(h, 0),
            'Coarse TR iterations': sum(
                count for level, count in counts.items()
                if level < h
            ),
            'Outer stages': len(out.history),
            'Final smoothed F': last.get('F_mu', np.nan),
            'Final stationarity': last.get(
                'stationarity_upper', np.nan
            )
        }

    a = summarize(single, out_single, time_single)
    b = summarize(multi, out_multi, time_multi)

    print('\n========== COMPARISON ==========')
    print(f"{'Metric':<25} {'Single TR':>18} {'Multilevel':>18}")
    print('-' * 65)

    for key in a:
        print(
            f"{key:<25} "
            f"{str(a[key]):>18} "
            f"{str(b[key]):>18}"
        )

    print('\nMultilevel recursion counters:', multi.counters)
    print('Multilevel steps by level:',
          dict(Counter(row['level'] for row in multi.history)))
    print("\n========== STATIONARITY CHANGE BY STEP TYPE AT mu=0.005==========")

    if hasattr(multi, 'history_stationarity'):
        for mode in ('current', 'coarse'):
            rows = [
                r for r in multi.history_stationarity
                if r['mode'] == mode
                and np.isclose(r['mu'],0.0125)
                ]
            if not rows:
                continue

            
            changes = np.array([r['change'] for r in rows])

            print(f"\n{mode.upper()} steps")
            print(f"Accepted steps: {len(rows)}")
            print(f"Mean stationarity change: {np.mean(changes):.6e}")
            print(f"Median stationarity change: {np.median(changes):.6e}")
            print(f"Steps reducing stationarity: {np.sum(changes < 0)}")
            
    print("\n========== TRUST-REGION RADII and RADIUS RATIOS AT mu=0.05 ==========")

    for name, solver in [('Single', single), ('Multi', multi)]:
        fine = len(solver.levels) - 1

        rows = [
            h for h in solver.history
            if h['level'] == fine
            and h['mode'] == 'current'
            and h['accepted']
            and np.isclose(h['mu'], 0.0125)
            ]
        ratios = np.array([h['step_ratio'] for h in rows])

        #radii = np.array([h['delta'] for h in rows])

        if len(ratios):
            print(
                f"{name}: "
                f"mean={np.mean(ratios):.3e}, "
                f"median={np.median(ratios):.3e}, "
                f"near_boundary={np.mean(ratios>=0.99):.1%}"
            )
    print("\n========== STEPS BY LEVEL AND MODE ==========")

    print(dict(Counter((row['level'], row['mode'])for row in multi.history)))

    print("\n========== ACCEPTED STEPS BY LEVEL AND MODE ==========")

    print(dict(Counter((row['level'], row['mode']) for row in multi.history if row['accepted'])))
    
    print("\n========== COARSE ELIGIBILITY BY OUTER STAGE ==========")

    fine = len(multi.levels) - 1

    for stage in out_multi.history:
        mu = stage['mu']

        rows = [h for h in multi.history if h['level'] == fine and np.isclose(h['mu'], mu)]

        eligible = sum(h['eligible'] for h in rows)
        coarse = sum(h['mode'] == 'coarse' for h in rows)
        accepted_coarse = sum(
            h['mode'] == 'coarse' and h['accepted']
            for h in rows
        )

        print(
            f"mu={mu:.5f}: "
            f"iterations={len(rows)}, "
            f"eligible={eligible}, "
            f"coarse={coarse}, "
            f"accepted_coarse={accepted_coarse}"
        )
        
    for name, solver in [('Single', single), ('Multi', multi)]:
        print(f"\n{name} computational costs:")
        for ell, d in solver.cost.items():
            print(
                f"Level {ell}: "
                f"evaluate {d['eval_calls']} calls, "
                f"{d['eval_time']:.2f} s; "
                f"objective {d['obj_calls']} calls, "
                f"{d['obj_time']:.2f} s"
            )
        

    # ==========================================
    # 4. Comparison plots in Spyder
    # ==========================================
    if not args.no_plots:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        for out, label in [
            (out_single, 'Single-level TR'),
            (out_multi, 'Multilevel IRMTR')
        ]:
            stages = [row['j'] for row in out.history]

            axes[0].plot(
                stages,
                [row['F_mu'] for row in out.history],
                'o-', label=label
            )

            axes[1].semilogy(
                stages,
                [row['stationarity_upper']
                 for row in out.history],
                'o-', label=label
            )

        axes[0].set_title('Smoothed objective')
        axes[0].set_xlabel('Outer smoothing iteration')
        axes[0].set_ylabel(r'$F_{\mu_j}$')

        axes[1].set_title('Original proximal stationarity')
        axes[1].set_xlabel('Outer smoothing iteration')
        axes[1].set_ylabel('Stationarity estimate')
        axes[1].axhline(
            outer_args['eps'],
            linestyle='--',
            color='gray',
            label='Tolerance'
        )

        for ax in axes:
            ax.grid(True, alpha=.3)
            ax.legend()

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
