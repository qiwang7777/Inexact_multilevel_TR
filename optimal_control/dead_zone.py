"""P1-FEM discretization of a 2-D elliptic optimal control dead-zone problem.

All control nodes, including boundary nodes, are unknowns. Only the *state*
uses a homogeneous Dirichlet boundary condition. The nonsmooth integral is
approximated by nodal mass lumping so that its W-metric prox is separable.

Dependencies: NumPy and SciPy.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class Parameters:
    alpha: float = 1e-4
    beta: float = 1e-2
    c: float = 2.0


def mesh(n: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Unit square split into n*n cells, 2 CCW triangles per cell.

    The diagonal runs from bottom-left to top-right on every square.
    Vertices are indexed as ix + (n+1)*iy, including the boundary.
    """
    if n < 2 or n % 2:
        raise ValueError("n must be even and >= 2")
    x, y = np.meshgrid(np.arange(n + 1) / n, np.arange(n + 1) / n)
    vertices = np.column_stack((x.ravel(), y.ravel()))
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    bl = (i + (n + 1) * j).ravel()
    br, tl, tr = bl + 1, bl + (n + 1), bl + n + 2
    triangles = np.empty((2 * n * n, 3), dtype=np.int64)
    triangles[0::2] = np.column_stack((bl, br, tr))
    triangles[1::2] = np.column_stack((bl, tr, tl))
    return vertices, triangles


def fem_matrices(vertices: NDArray, triangles: NDArray) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Exact P1 stiffness+reaction matrix K and consistent mass M.

    a(x)=1+x_1/2 is affine; barycenter quadrature integrates a*gradphi_i
    dot gradphi_j exactly because gradients are constant per triangle.
    The element mass matrix is |T|/12 * [[2,1,1],[1,2,1],[1,1,2]].
    """
    xy = vertices[triangles]
    dx1 = xy[:, 1] - xy[:, 0]
    dx2 = xy[:, 2] - xy[:, 0]
    twice_area = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    if np.any(twice_area <= 0):
        raise ValueError("Mesh contains non-CCW or degenerate triangles")
    area = 0.5 * twice_area
    # Gradients of the three affine basis functions on each triangle.
    gx = np.stack((xy[:, 1, 1] - xy[:, 2, 1],
                   xy[:, 2, 1] - xy[:, 0, 1],
                   xy[:, 0, 1] - xy[:, 1, 1]), axis=1) / twice_area[:, None]
    gy = np.stack((xy[:, 2, 0] - xy[:, 1, 0],
                   xy[:, 0, 0] - xy[:, 2, 0],
                   xy[:, 1, 0] - xy[:, 0, 0]), axis=1) / twice_area[:, None]
    a_bar = 1.0 + 0.5 * xy[:, :, 0].mean(axis=1)
    local_mass = (area / 12.0)[:, None, None] * np.array(
        [[2., 1., 1.], [1., 2., 1.], [1., 1., 2.]])[None, :, :]
    local_k = (area * a_bar)[:, None, None] * (
        gx[:, :, None] * gx[:, None, :] + gy[:, :, None] * gy[:, None, :]
    ) + local_mass
    rows = np.broadcast_to(triangles[:, :, None], (len(triangles), 3, 3)).ravel()
    cols = np.broadcast_to(triangles[:, None, :], (len(triangles), 3, 3)).ravel()
    N = len(vertices)
    K = sparse.coo_matrix((local_k.ravel(), (rows, cols)), shape=(N, N)).tocsr()
    M = sparse.coo_matrix((local_mass.ravel(), (rows, cols)), shape=(N, N)).tocsr()
    return K, M


def prolongation(n_coarse: int) -> sparse.csr_matrix:
    """Nodal P1 interpolation (n_coarse -> 2*n_coarse).

    At odd/odd fine nodes, use the *two* endpoints of the SW-NE diagonal,
    not the four bilinear interpolant corners.
    """
    if n_coarse < 2:
        raise ValueError("n_coarse must be >= 2")
    n_fine = 2 * n_coarse
    fine_n = (n_fine + 1)**2
    coarse_width = n_coarse + 1
    ii, jj = np.meshgrid(np.arange(n_fine + 1), np.arange(n_fine + 1))
    ii, jj = ii.ravel(), jj.ravel()
    fine_ids = np.arange(fine_n, dtype=np.int64)
    ci, cj = ii // 2, jj // 2
    idx00 = ci + coarse_width*cj
    is_i_odd, is_j_odd = (ii % 2).astype(bool), (jj % 2).astype(bool)
    # Non-corner nodes get two 1/2 weights. At a square center the
    # midpoint belongs to the (bottom-left, top-right) diagonal.
    col2 = idx00 + np.where(is_i_odd, 1, coarse_width)
    both = is_i_odd & is_j_odd
    col2[both] = idx00[both] + coarse_width + 1
    corner = ~is_i_odd & ~is_j_odd
    indices = np.concatenate((idx00, col2[~corner]))
    rowids = np.concatenate((fine_ids, fine_ids[~corner]))
    values = np.concatenate((np.where(corner, 1.0, .5),
                             np.full(np.count_nonzero(~corner), .5)))
    P = sparse.coo_matrix((values, (rowids, indices)),
        shape=(fine_n, coarse_width**2)).tocsr()
    return P


def restriction(P: sparse.csr_matrix, w_coarse: NDArray, w_fine: NDArray) -> sparse.csr_matrix:
    """Mass-weighted adjoint R=W_c^{-1} P.T W_f (NOT plain P.T)."""
    if P.shape != (len(w_fine), len(w_coarse)):
        raise ValueError("P and lumped weights have inconsistent shapes")
    return (sparse.diags(1.0/w_coarse) @ P.T @ sparse.diags(w_fine)).tocsr()


def local_transfer_example() -> tuple[NDArray, NDArray]:
    """The exact 6-by-3 and 3-by-6 local matrices from the problem statement."""
    P = np.array([[1,0,0], [0,1,0], [0,0,1],
                  [.5,.5,0], [0,.5,.5], [.5,0,.5]], float)
    # Coarse weights A/3, fine weights A/12 at corners, A/4 at midpoints.
    R = (P.T * np.array([1/4,1/4,1/4,3/4,3/4,3/4])[None, :])
    return P, R


class DeadZoneProblem:
    """Discrete reduced objective at a single mesh resolution.

    f(z)=0.5*(y_h-y_d,h)^T M (y_h-y_d,h) + alpha/2*z^T M z
    phi(z)=beta*sum_i w_i*max(|z_i|-c,0), w=M*1.

    State: K_II*y_I = M_{I,:}*z; y_boundary=0.
    Note: phi uses *mass lumping*, not the exact integral of the P1 hinge.
    """
    def __init__(self, n: int, params: Parameters | None = None):
        self.n = n
        self.params = params or Parameters()
        self.vertices, self.triangles = mesh(n)
        self.K, self.M = fem_matrices(self.vertices, self.triangles)
        self.w = np.asarray(self.M.sum(axis=1)).ravel()
        coords = self.vertices
        self.yd = 2.0 * np.sin(2*np.pi*coords[:,0]) * np.sin(np.pi*coords[:,1])
        tol = 1e-12
        interior = ((coords[:,0] > tol) & (coords[:,0] < 1.-tol) &
                    (coords[:,1] > tol) & (coords[:,1] < 1.-tol))
        self.I = np.flatnonzero(interior)
        self.A = self.K[self.I, :][:, self.I].tocsc()
        self.B = self.M[self.I, :].tocsr()
        self._LU = splu(self.A)

    @property
    def N(self) -> int:
        return len(self.w)

    def _validate(self, z: NDArray) -> NDArray:
        arr = np.asarray(z, float).reshape(-1)
        if arr.size != self.N:
            raise ValueError(f"Expected {self.N} control coefficients, got {arr.size}")
        return arr

    def state(self, z: NDArray) -> NDArray:
        """Full nodal state, including the zero Dirichlet boundary."""
        z = self._validate(z)
        y = np.zeros(self.N)
        y[self.I] = self._LU.solve(self.B @ z)
        return y

    def adjoint(self, y: NDArray) -> NDArray:
        """Interior adjoint p solves A^T*p = [M(y-yd)]_I."""
        r = self.M @ (self._validate(y) - self.yd)
        return self._LU.solve(r[self.I], trans='T')

    def smooth_value_gradient(self, z: NDArray) -> tuple[float, NDArray]:
        z = self._validate(z)
        y = self.state(z)
        r = y - self.yd
        f = (0.5 * float(r @ (self.M @ r)) +
             self.params.alpha/2 * float(z @ (self.M @ z)))
        p = self._LU.solve((self.M @ r)[self.I], trans='T')
        g = self.B.T @ p + self.params.alpha * (self.M @ z)
        return f, np.asarray(g).ravel()

    def smooth_value(self, z: NDArray) -> float:
        z = self._validate(z)
        r = self.state(z) - self.yd
        return 0.5 * float(r @ (self.M @ r)) + self.params.alpha/2 * float(z @ (self.M @ z))

    def smooth_hessp(self, v: NDArray) -> NDArray:
        """Matrix-free exact reduced Hessian-vector product (Euclidean)."""
        v = self._validate(v)
        dy_i = self._LU.solve(self.B @ v)
        dp_i = self._LU.solve(self.M[self.I,:][:,self.I] @ dy_i, trans='T')
        return np.asarray(self.B.T @ dp_i + self.params.alpha*(self.M @ v)).ravel()

    def penalty(self, z: NDArray) -> float:
        z = self._validate(z)
        return self.params.beta * float(self.w @ np.maximum(np.abs(z)-self.params.c, 0.0))

    def objective(self, z: NDArray) -> float:
        return self.smooth_value(z) + self.penalty(z)

    def prox_w(self, u: NDArray, step: float) -> NDArray:
        """prox^W_{step*phi}(u) in W-diagonal metric.

        For s=|u_i|: s<=c => s; c<s<=c+step*beta => c;
        s>c+step*beta => s-step*beta. The threshold is INDEPENDENT of w_i
        because both the penalty and the proximal metric use w_i.
        """
        if step <= 0:
            raise ValueError("step must be positive")
        u = self._validate(u)
        s = np.abs(u)
        new_s = np.where(s <= self.params.c, s,
                     np.maximum(self.params.c, s - step*self.params.beta))
        return np.copysign(new_s, u)

    def stationarity(self, z: NDArray, L: float) -> float:
        """W-norm of proximal gradient mapping, zero at a discrete optimum."""
        _, g = self.smooth_value_gradient(z)
        trial = self.prox_w(z - (g/self.w)/L, 1.0/L)
        d = L*(z-trial)
        return float(np.sqrt(self.w @ (d*d)))


class Hierarchy:
    """Levels are ordered from coarse to fine. Allows other power-of-2 sizes for tests."""
    def __init__(self, resolutions: tuple[int, ...] = (32,64,128),
                 params: Parameters | None = None):
        if not resolutions or any(b != 2*a for a,b in zip(resolutions, resolutions[1:])):
            raise ValueError("Mesh resolutions must double at each level")
        self.levels = [DeadZoneProblem(n, params) for n in resolutions]
        self.P = [prolongation(n) for n in resolutions[:-1]]
        self.R = [restriction(P, a.w, b.w)
                  for P, a,b in zip(self.P, self.levels[:-1], self.levels[1:])]


@dataclass
class Result:
    x: NDArray
    objective: float
    stationarity: float
    iterations: int
    history: list[tuple[int, float, float]]
    L: float


def estimate_L(problem: DeadZoneProblem, steps: int = 12) -> float:
    """W-operator norm estimate for W^{-1} Hessian; backtracking ensures safety."""
    rng = np.random.default_rng(100)
    v = rng.standard_normal(problem.N)
    v /= np.sqrt(problem.w @ (v*v))
    L = 0.0
    for _ in range(steps):
        Av = problem.smooth_hessp(v) / problem.w
        nrm = float(np.sqrt(problem.w @ (Av*Av)))
        if not np.isfinite(nrm) or nrm == 0.0:
            raise ArithmeticError("Invalid Hessian norm")
        v = Av/nrm
        L = float(v @ problem.smooth_hessp(v))
    return max(1.5*L, problem.params.alpha)


def solve_fista(problem: DeadZoneProblem, x0: NDArray | None = None,
                maxiter: int = 300, rtol: float = 1e-7,
                L: float | None = None, verbose: bool = False) -> Result:
    """Monotone FISTA with W-metric proximal map and quadratic backtracking.

    This is a runnable validation/baseline solver, NOT an implementation of
    the linked repository's inexact multilevel trust-region algorithm.
    """
    x = np.zeros(problem.N) if x0 is None else problem._validate(x0).copy()
    q = x.copy()
    t = 1.0
    L = estimate_L(problem) if L is None else float(L)
    if L <= 0:
        raise ValueError("L must be positive")
    Fx = problem.objective(x)
    history: list[tuple[int, float, float]] = []
    for it in range(1, maxiter+1):
        restarted = False
        while True:
            fy, gy = problem.smooth_value_gradient(q)
            # If the accelerated trial raises F, restart from x.
            for _ in range(80):
                xnew = problem.prox_w(q - (gy / problem.w)/L, 1/L)
                d = xnew-q
                fn = problem.smooth_value(xnew)
                majorant = fy + float(gy @ d) + 0.5*L*float(problem.w @ (d*d))
                if fn <= majorant + 1e-12:
                    break
                L *= 2.0
            else:
                raise RuntimeError("Backtracking did not find a majorant")
            Fnew = fn + problem.penalty(xnew)
            if Fnew <= Fx + 1e-12 or restarted:
                break
            restarted = True
            q = x.copy()
            t = 1.0
        if Fnew > Fx + 1e-10:
            raise ArithmeticError("Proximal descent failed after momentum restart")
        if restarted:
            next_t = 1.0
            qnext = xnew.copy()
        else:
            next_t = (1.0 + np.sqrt(1.0+4.0*t*t))/2.0
            qnext = xnew + ((t-1.0)/next_t)*(xnew-x)
        x, q, t, Fx = xnew, qnext, next_t, Fnew
        # At a fixed cost, stop on the weighted gradient-mapping residual.
        residual = problem.stationarity(x, L)
        history.append((it, Fx, residual))
        if verbose and (it == 1 or it % 10 == 0 or residual <= rtol):
            print(f"  iteration {it:4d}: objective={Fx:.9e}, residual={residual:.3e}, L={L:.3e}", flush=True)
        if residual <= rtol:
            break
    return Result(x, Fx, residual, it if maxiter else 0, history, L)


def solve_nested(hierarchy: Hierarchy, maxiter: int = 300,
                 rtol: float = 1e-7, verbose: bool = True) -> list[Result]:
    """Solve on coarse grid, prolongate, then refine: a multilevel warm start.

    Does not claim to be an inexact multilevel trust-region method.
    """
    results: list[Result] = []
    x0 = None
    for ell, problem in enumerate(hierarchy.levels):
        if verbose:
            print(f"Level {ell}: {problem.n}x{problem.n}, control DOF={problem.N}, "
                  f"state DOF={len(problem.I)}", flush=True)
        if ell:
            x0 = np.asarray(hierarchy.P[ell-1] @ results[-1].x).ravel()
        result = solve_fista(problem, x0, maxiter, rtol, verbose=verbose)
        results.append(result)
    return results
