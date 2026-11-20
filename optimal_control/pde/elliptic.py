"""Example 2: P1 state/P1 control for variable-diffusion linear elliptic PDE.

  -div((1+x_1/2) grad y) + y = z, y|boundary = 0
  y_d = 2 sin(2 pi x_1) sin(pi x_2).

The state/adjoint PCG solves and residual-based gradient error certificate
come directly from the supplied Example 2; FE geometry and quadrature are
shared with the semilinear example.
"""
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ..geometry import create_unit_square_mesh, element_geometry, triangle_quadrature

@dataclass
class FEMLevel:
    n: int
    nodes: np.ndarray
    triangles: np.ndarray
    interior: np.ndarray
    M: sp.csr_matrix
    W: np.ndarray
    A: sp.csr_matrix
    factor: object
    yd_load: np.ndarray
    yd_square: float


def mesh_level(n: int) -> FEMLevel:
    """Linear PDE with P1 state and P1 control on the SAME triangular mesh."""
    if n < 2:
        raise ValueError('n>=2 is required')
    base = create_unit_square_mesh(n)
    nodes, E = base.nodes, base.elements
    pts = nodes[E]
    area, grads = element_geometry(base)
    mass_local = (area[:, None, None] / 12.0) * np.array(
        [[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float)
    # a(x)=1+x_1/2; exact element average for affine coefficient
    diff_coeff = 1.0 + .5 * pts[:, :, 0].mean(axis=1)
    stiff_local = (area * diff_coeff)[:, None, None] * np.einsum(
        'eik,ejk->eij', grads, grads)
    shape = (len(nodes), len(nodes))
    rows = np.repeat(E, 3, axis=1).ravel()
    cols = np.tile(E, (1, 3)).ravel()
    M = sp.coo_matrix((mass_local.ravel(), (rows, cols)), shape=shape).tocsr()
    A = sp.coo_matrix(((stiff_local + mass_local).ravel(),
                       (rows, cols)), shape=shape).tocsr()
    W = np.asarray(M.sum(axis=1)).ravel()
    bary, weights = triangle_quadrature()
    pq = np.einsum('eic,qi->eqc', pts, bary)
    ydq = 2.0 * np.sin(2.0*np.pi*pq[:,:,0]) * np.sin(np.pi*pq[:,:,1])
    local_load = area[:,None] * np.einsum('eq,q,qi->ei', ydq, weights, bary)
    load = np.bincount(E.ravel(), weights=local_load.ravel(), minlength=len(nodes))
    square = float(np.sum(area[:,None] * weights[None,:] * ydq**2))
    return FEMLevel(n, nodes, E, base.interior, M, W, A, None, load, square)


def state(mesh: FEMLevel,z:np.ndarray)->np.ndarray:
    # Reference-only solve, used by optional verification routines.
    if mesh.factor is None:
        mesh.factor=spla.splu(mesh.A[mesh.interior,:][:,mesh.interior].tocsc())
    rhs=np.asarray(mesh.M @ z).ravel()
    y=np.zeros(len(mesh.nodes))
    y[mesh.interior]=mesh.factor.solve(rhs[mesh.interior])
    return y


def adjoint(mesh: FEMLevel,y:np.ndarray)->np.ndarray:
    # Reference-only solve, used by optional verification routines.
    if mesh.factor is None:
        mesh.factor=spla.splu(mesh.A[mesh.interior,:][:,mesh.interior].tocsc())
    # p solves A p = (y_d-y,phi), p=0 at boundary.
    rhs=mesh.yd_load-np.asarray(mesh.M@y).ravel()
    p=np.zeros(len(mesh.nodes))
    p[mesh.interior]=mesh.factor.solve(rhs[mesh.interior])
    return p


def smooth_value(mesh: FEMLevel,z:np.ndarray,y:np.ndarray,alpha:float)->float:
    # Integral of (y_h-y_d)^2/2, using exact P1 mass and target quadrature.
    My=np.asarray(mesh.M@y).ravel()
    Mz=np.asarray(mesh.M@z).ravel()
    return float(.5*y@My - mesh.yd_load@y + .5*mesh.yd_square
                 +.5*alpha*z@Mz)


def smooth_gradient(mesh: FEMLevel,z:np.ndarray,p:np.ndarray,alpha:float)->np.ndarray:
    # Euclidean derivative = M (alpha z - p); return W-Riesz gradient.
    return np.asarray(mesh.M@(alpha*z-p)).ravel()/mesh.W


# Backwards-compatible alias for the P1 control-space transfers.
from ..transfer_operators import p1_transfers as transfers


def inner(mesh,u,v):return float(np.dot(mesh.W*u,v))


def norm(mesh,u):return np.sqrt(max(0.,inner(mesh,u,u)))


# Residual-to-gradient-error coefficient, inherited from Example 2:
# A_II >= (1+2*pi^2) M_II >= ((1+2*pi^2)/4) W_II.
RESIDUAL_CONSTANT = 4.0/(1.0 + 2.0*np.pi**2)


class PDEMultigrid:
    """Symmetric geometric V-cycle PCG preconditioner for the state operator.

    Level 0 is solved directly; all state/adjoint solves on finer levels use
    preconditioned CG with a strict residual-based stopping criterion.
    Galerkin transfers for the assembled *linear PDE* use P.T, not the
    mass-weighted optimization restriction R=Wc^-1 P.T Wf.
    """
    def __init__(self, meshes, sweeps=2, omega=.65):
        self.meshes=meshes
        self.A=[m.A[m.interior,:][:,m.interior].tocsr() for m in meshes]
        self.Dinv=[1./a.diagonal() for a in self.A]
        self.P=[None]
        for ell in range(1,len(meshes)):
            _,pf=transfers(meshes[ell-1],meshes[ell])
            self.P.append(pf[meshes[ell].interior,:][:,meshes[ell-1].interior].tocsr())
        self.base_factor=spla.splu(self.A[0].tocsc())
        self.omega=float(omega)
        self.sweeps=int(sweeps)
        self.operators=[]
        for ell,a in enumerate(self.A):
            self.operators.append(spla.LinearOperator(a.shape,matvec=lambda b,ell=ell:self.vcycle(ell,b),dtype=float))
        self.counts={'state_cg_iterations':0,'adjoint_cg_iterations':0,
                     'state_solves':0,'adjoint_solves':0,'cg_failures':0,
                     'gradient_refinements':0}

    def vcycle(self,ell,b):
        if ell==0:return self.base_factor.solve(b)
        a=self.A[ell];di=self.Dinv[ell];P=self.P[ell]
        x=np.zeros_like(b)
        for _ in range(self.sweeps):
            x+=self.omega*di*(b-a@x)
        r=b-a@x
        x+=P@self.vcycle(ell-1,P.T@r)
        for _ in range(self.sweeps):
            x+=self.omega*di*(b-a@x)
        return x

    def dual_residual_norm(self,ell,r):
        w=self.meshes[ell].W[self.meshes[ell].interior]
        return float(np.sqrt(np.dot(r/w,r)))

    def pcg(self,ell,b,r_target,kind,x0=None):
        a=self.A[ell]
        if np.linalg.norm(b)==0.:
            self.counts[kind+'_solves']+=1
            return np.zeros_like(b),0.
        if x0 is None:x0=np.zeros_like(b)
        start_r=b-a@x0
        nr=self.dual_residual_norm(ell,start_r)
        if nr<=r_target:
            self.counts[kind+'_solves']+=1
            return x0.copy(),nr
        # Residual in the actual weighted dual norm is verified independently
        # of scipy.cg's internal stopping test; tighten if necessary.
        x=x0.copy()
        for rep in range(7):
            # scipy.cg uses Euclidean ||b|| to scale rtol, whereas our
            # certificate measures ||r||_{W_I^-1}. Bound by
            # ||r||_{W_I^-1} <= ||r||_2/sqrt(min W_I).
            min_w=float(np.min(self.meshes[ell].W[self.meshes[ell].interior]))
            rel=min(.3,max(1e-14,.50*r_target*np.sqrt(min_w)/max(np.linalg.norm(b),1e-300)))
            iterations=[0]
            def callback(v):iterations[0]+=1
            x,info=spla.cg(a,b,x0=x,M=self.operators[ell],rtol=rel,atol=0.,
                           maxiter=300,callback=callback)
            self.counts[kind+'_cg_iterations']+=iterations[0]
            residual=b-a@x
            new_norm=self.dual_residual_norm(ell,residual)
            if np.isfinite(new_norm) and new_norm<=r_target:
                self.counts[kind+'_solves']+=1
                return x,new_norm
            nr=new_norm
            if info<0:break
        self.counts['cg_failures']+=1
        raise RuntimeError(f'CG failed to certify {kind} solve on level {ell}: '
                           f'||residual||_W^-1={nr:.3e}, target={r_target:.3e}')

    def state(self,ell,z):
        m=self.meshes[ell]
        rhs=np.asarray(m.M@z).ravel()[m.interior]
        rhs_norm=self.dual_residual_norm(ell,rhs)
        # Tight PDE state solves to keep trust-region objective values accurate.
        target=2e-11*max(rhs_norm,1e-12)
        yi,ry=self.pcg(ell,rhs,target,'state')
        y=np.zeros(len(m.nodes));y[m.interior]=yi
        return y,ry

    def adjoint(self,ell,y,e_target,y_residual_norm,p0=None):
        m=self.meshes[ell]
        rhs=np.asarray(m.yd_load-m.M@y).ravel()[m.interior]
        # The gradient error bound has two terms: C||rp|| + C²||ry||.
        state_part=RESIDUAL_CONSTANT**2*y_residual_norm
        if state_part>=e_target:
            raise RuntimeError(f'State residual error ({state_part:g}) exceeds gradient budget ({e_target:g})')
        rp_target=max(1e-14, (e_target-state_part)/RESIDUAL_CONSTANT * .75)
        pi,rp=self.pcg(ell,rhs,rp_target,'adjoint',x0=p0)
        p=np.zeros(len(m.nodes));p[m.interior]=pi
        e=RESIDUAL_CONSTANT*rp+state_part
        return p,e,pi



class ExactReducedHessian:
    """Matrix-free exact *discrete* P1-control reduced Hessian in W metric.

    State A_II y_I=(M z)_I, adjoint A_II p_I=(yd_load-M y)_I.
    A and M are symmetric. Hence, for s in the full P1 control space,
      dy_I=A_II^-1 (M s)_I,
      dp_I=-A_II^-1 (M dy)_I,
      H_f s = W^-1 M (alpha*s - dp).
    The same direct factorization is reused across model builds since A is
    linear and z-independent. Its approximation error is floating point only.
    """
    def __init__(self, mesh, alpha, stats=None):
        self.mesh = mesh
        self.alpha = float(alpha)
        self.stats = stats
        if self.stats is not None:
            self.stats['builds'] += 1
        if mesh.factor is None:
            mesh.factor = spla.splu(mesh.A[mesh.interior,:][:,mesh.interior].tocsc())
        self.factor = mesh.factor
        self.bound = self.alpha + RESIDUAL_CONSTANT**2
        # Low-alpha baseline is a safe SPD preconditioner; exact spectrum
        # has the alpha*M/W contribution plus the nonnegative tracking part.
        self.preconditioner_diagonal = np.full(len(mesh.W), max(alpha, 1e-3))

    def matvec(self, s):
        if self.stats is not None:
            self.stats['products'] += 1
        m = self.mesh
        I = m.interior
        dy = np.zeros(len(m.nodes))
        dy[I] = self.factor.solve(np.asarray(m.M@s).ravel()[I])
        dp = np.zeros(len(m.nodes))
        dp[I] = self.factor.solve(-np.asarray(m.M@dy).ravel()[I])
        return np.asarray(m.M@(self.alpha*s - dp)).ravel()/m.W
