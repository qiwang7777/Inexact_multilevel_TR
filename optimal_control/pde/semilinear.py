"""Semilinear state -Delta u + u**3=z, adjoint and exact reduced Hessian."""

import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ..geometry import (element_geometry, triangle_quadrature,
                        assemble_control_load, get_stiffness)


def _nonlinear_fe_cache(mesh):
    """Cache FE quadrature and sparse Jacobian structure."""

    cache = getattr(mesh, '_nonlinear_fe_cache', None)

    if cache is not None:
        return cache

    E = mesh.elements
    n = len(mesh.nodes)

    bary, weights = triangle_quadrature()

    wb = weights[:, None] * bary

    wbb = (
        weights[:, None, None]
        * bary[:, :, None]
        * bary[:, None, :]
    )

    rows = np.repeat(E, 3, axis=1).ravel()
    cols = np.tile(E, (1, 3)).ravel()

    key = rows.astype(np.int64) * n + cols

    keys, inverse = np.unique(key, return_inverse=True)

    indptr = np.concatenate((
        [0],
        np.cumsum(
            np.bincount(keys // n, minlength=n),
            dtype=np.int64
        )
    ))

    csr_indices = (keys % n).astype(np.int32)

    cache = (
        wb,
        wbb,
        inverse,
        csr_indices,
        indptr,
        len(keys)
    )

    mesh._nonlinear_fe_cache = cache

    return cache


def nonlinear_load(mesh, u):
    """Compute (u^3, phi) without assembling the Jacobian."""

    areas, _ = element_geometry(mesh)

    wb, _, _, _, _, _ = _nonlinear_fe_cache(mesh)

    E = mesh.elements

    bary, _ = triangle_quadrature()

    uq = u[E] @ bary.T

    nloc = areas[:, None] * (uq**3 @ wb)

    return np.bincount(
        E.ravel(),
        weights=nloc.ravel(),
        minlength=len(mesh.nodes)
    )


def assemble_nonlinear_terms(mesh, u):
    """Assemble nonlinear residual and Jacobian using cached structure."""

    areas, _ = element_geometry(mesh)

    wb, wbb, inverse, cols, indptr, nnz = (
        _nonlinear_fe_cache(mesh)
    )

    E = mesh.elements

    bary, _ = triangle_quadrature()

    uq = u[E] @ bary.T

    # Nonlinear load
    nloc = areas[:, None] * (uq**3 @ wb)

    N = np.bincount(
        E.ravel(),
        weights=nloc.ravel(),
        minlength=len(mesh.nodes)
    )

    # Numerical Jacobian values
    jloc = areas[:, None, None] * np.einsum(
        'eq,qij->eij',
        3.0 * uq**2,
        wbb,
        optimize=True
    )

    # Assemble directly into cached CSR structure
    jdata = np.bincount(
        inverse,
        weights=jloc.ravel(),
        minlength=nnz
    )

    J = sp.csr_matrix(
        (jdata, cols, indptr),
        shape=(len(mesh.nodes), len(mesh.nodes))
    )

    return N, J


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


def value_f(mesh, z, u, alpha=1e-4, w=-1.0):
    areas, _ = element_geometry(mesh)
    bary, weights = triangle_quadrature()
    uq = u[mesh.elements] @ bary.T
    return (0.5 * np.sum(areas[:, None] * weights[None, :] * (uq-w)**2)
            + 0.5 * alpha * np.dot(areas, z*z))


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


