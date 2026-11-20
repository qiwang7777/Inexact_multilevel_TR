"""Limited-memory, Galerkin, and Moreau-augmented Hessian models."""

import numpy as np
from .geometry import p0_inner


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


