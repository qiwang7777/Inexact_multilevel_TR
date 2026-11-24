"""Obstacle problem backend for the repository's ORIGINAL IRMTRMoreau solver.

On each full P1 nodal space (natural Neumann boundary conditions):
  f(u) = (1/2) u.T A u - b.T u,
  A = stiffness(a) + consistent_mass,
  a = 1+4 sin(2*pi*x1)^2 sin(2*pi*x2)^2,
  b_i = integral(g*phi_i) by seven-point triangle quadrature,
  phi(u) = I_{u>=psi},
where g=10cos(pi*x1)cos(pi*x2) and
psi=0.1exp(-40((x1-.5)^2+(x2-.5)^2))-.05.

The exact W-Hessian of f is W^-1 A. The Moreau envelope of phi is
(2mu)^-1 ||min(u-psi,0)||^2_W.  Coarse Hessians are inherited via
GalerkinHessian(R B_f P) by IRMTRMoreau, not constructed here.
"""
from __future__ import annotations

import time
import numpy as np

from .types import Evaluation
from .hessian_models import HessianPlusDiagonal, LBFGSHessian
from .obstacle_regularization import (
    obstacle_prox,
    obstacle_moreau_value, obstacle_moreau_grad,
    obstacle_moreau_hessian_diagonal,
)


class ExactObstacleHessian:
    """Matrix-free exact Hessian W^-1 A, W-self-adjoint and positive definite."""
    def __init__(self, mesh, stats=None):
        self.mesh = mesh
        self.stats = stats
        row_abs = np.asarray(abs(mesh.A).sum(axis=1)).ravel()
        self.bound = float(np.max(row_abs / mesh.W))
        self.preconditioner_diagonal = (
            np.asarray(mesh.A.diagonal(), dtype=float).ravel() / mesh.W)
        if stats is not None:
            stats['builds'] += 1

    def matvec(self, v):
        if self.stats is not None:
            self.stats['products'] += 1
        return np.asarray(self.mesh.A @ np.asarray(v, dtype=float)).ravel() / self.mesh.W


class ObstacleBackend:
    """Backend protocol used by optimal_control.irmtr.IRMTRMoreau."""
    control_space = 'p1'

    def __init__(self, meshes):
        self.data = {}
        for mesh in meshes:
            self.data[id(mesh)] = dict(psi=mesh.psi.copy(), b=mesh.load.copy())
        self.pde = None  # This is a direct variational problem, not PDE control.

    def _data(self, mesh):
        return self.data[id(mesh)]

    def obstacle(self, mesh):
        return self._data(mesh)['psi']

    @staticmethod
    def inner(mesh, x, y):
        return float(np.dot(mesh.W * np.asarray(x), np.asarray(y)))

    @staticmethod
    def norm(mesh, x):
        return float(np.sqrt(max(0., ObstacleBackend.inner(mesh, x, x))))

    def smooth_value(self, mesh, z):
        b = self._data(mesh)['b']
        return float(0.5 * np.dot(z, mesh.A @ z) - np.dot(b, z))

    def smooth_gradient(self, mesh, z):
        b = self._data(mesh)['b']
        return np.asarray(mesh.A @ z - b).ravel() / mesh.W

    def objective(self, solver, ell, z, mu, return_state=False, u_initial=None):
        t0 = time.perf_counter()
        mesh = solver.levels[ell].mesh
        psi = self.obstacle(mesh)
        if not np.all(np.isfinite(z)):
            value = np.inf
        else:
            value = self.smooth_value(mesh, z)
            if not solver.cfg.smooth_only:
                value += obstacle_moreau_value(mesh.W, z, psi, mu)
        solver.cost[ell]['obj_calls'] += 1
        solver.cost[ell]['obj_time'] += time.perf_counter() - t0
        return (float(value), None) if return_state else float(value)

    def evaluate(self, solver, ell, z, mu, tau, state=None,
                 u_initial=None, p_initial=None, delta=None):
        t0 = time.perf_counter()
        mesh = solver.levels[ell].mesh
        psi = self.obstacle(mesh)
        g = self.smooth_gradient(mesh, z)
        G = g if solver.cfg.smooth_only else g + obstacle_moreau_grad(z, psi, mu)
        F = self.smooth_value(mesh, z)
        if not solver.cfg.smooth_only:
            F += obstacle_moreau_value(mesh.W, z, psi, mu)
        solver.gradient_evals += 1
        solver.cost[ell]['eval_calls'] += 1
        solver.cost[ell]['eval_time'] += time.perf_counter() - t0
        # The gradient is obtained by sparse matvec; no PDE residual error.
        return Evaluation(float(F), g, G, 0.0, None, None, None)

    def make_model(self, solver, ell, z, mu, ev=None):
        mesh = solver.levels[ell].mesh
        if solver.cfg.hessian_mode == 'exact':
            H = ExactObstacleHessian(mesh, stats=solver.hessian_stats)
        elif solver.cfg.hessian_mode == 'lbfgs':
            H = LBFGSHessian(mesh.W, solver.cfg.curvature, solver.lbfgs_pairs[ell])
        else:
            raise ValueError('ObstacleBackend supports exact or lbfgs Hessians')
        if solver.cfg.smooth_only:
            return H
        return HessianPlusDiagonal(
            H, obstacle_moreau_hessian_diagonal(z, self.obstacle(mesh), mu))

    def original_stationarity(self, solver, z, r, eps_hint):
        """Original projected-gradient residual, evaluated at feasible point.

        The original objective is +infinity at infeasible smoothed iterates.
        The driver projects first and checks stationarity of the FEASIBLE point.
        """
        if r <= 0:
            raise ValueError('r must be positive')
        mesh = solver.levels[-1].mesh
        psi = self.obstacle(mesh)
        z_feas = obstacle_prox(z, psi)
        g = self.smooth_gradient(mesh, z_feas)
        projected_next = obstacle_prox(z_feas - r*g, psi)
        resid = self.norm(mesh, (z_feas - projected_next)/r)
        return float(resid), float(resid), 0.0
