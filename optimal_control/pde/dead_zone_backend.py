"""Dead-zone P1 elliptic backend plugged into the ORIGINAL IRMTRMoreau core.

The PDE, certified inexact gradient machinery, control metric and exact
reduced Hessian are the ones from run_elliptic.py.  Only the nonsmooth
penalty, Moreau terms, and original proximal residual are changed.
"""
from __future__ import annotations

import time
import numpy as np

from ..types import Evaluation
from ..hessian_models import LBFGSHessian, HessianPlusDiagonal
from ..dead_zone_regularization import (
    dead_zone_prox, dead_zone_moreau_value, dead_zone_moreau_grad,
    dead_zone_moreau_hessian_diagonal,
)
from .elliptic import norm, smooth_value, smooth_gradient, ExactReducedHessian
from .elliptic_backend import EllipticBackend


class DeadZoneEllipticBackend(EllipticBackend):
    """Pass backend=DeadZoneEllipticBackend(pde,c=2) to IRMTRMoreau."""
    control_space = 'p1'

    def __init__(self, pde, c=2.0):
        super().__init__(pde)
        if c < 0:
            raise ValueError('Dead-zone c must be >= 0')
        self.c = float(c)

    def objective(self, solver, ell, z, mu, return_state=False, u_initial=None):
        t0 = time.perf_counter()
        if not np.all(np.isfinite(z)):
            value, cache = np.inf, None
        else:
            m = solver.levels[ell].mesh
            y, ry = self.pde.state(self.pde_index(solver, ell), z)
            self.cached_states[ell] = (z.copy(), y.copy(), ry)
            value = smooth_value(m, z, y, solver.cfg.alpha)
            if not solver.cfg.smooth_only:
                value += dead_zone_moreau_value(m, z, mu, solver.cfg.beta, self.c)
            cache = (y, ry)
        solver.cost[ell]['obj_calls'] += 1
        solver.cost[ell]['obj_time'] += time.perf_counter() - t0
        return (value, cache) if return_state else value

    def evaluate(self, solver, ell, z, mu, tau, state=None,
                 u_initial=None, p_initial=None, delta=None):
        """Same certified gradient-forcing procedure as EllipticBackend.

        The forcing criterion must be tested against the *dead-zone*
        smoothed gradient G, rather than the original L1 smoothed G.
        """
        t0 = time.perf_counter()
        m = solver.levels[ell].mesh
        if state is not None:
            # Core caches either accepted trial state (y,residual) or (ev.u, ev.J).
            y, ry = state
            solver.cost[ell]['reused_trial_states'] += 1
            self.cached_states.pop(ell, None)
        else:
            cached = self.cached_states.pop(ell, None)
            if cached is not None and np.array_equal(cached[0], z):
                y, ry = cached[1], cached[2]
                solver.cache_hits += 1
                solver.cost[ell]['reused_trial_states'] += 1
            else:
                y, ry = self.pde.state(self.pde_index(solver, ell), z)

        if delta is None:
            delta = np.inf
        idx = self.pde_index(solver, ell)
        budget = solver.cfg.kappa_g * min(delta, max(tau, 1e-14))
        p0 = self.previous_adjoint[idx]
        if p_initial is not None:
            p0 = np.asarray(p_initial)[m.interior]

        for attempt in range(12):
            p, e, pi = self.pde.adjoint(idx, y, budget, ry, p0=p0)
            g = smooth_gradient(m, z, p, solver.cfg.alpha)
            G = (g if solver.cfg.smooth_only else
                 g + dead_zone_moreau_grad(z, mu, solver.cfg.beta, self.c))
            ng = norm(m, G)
            limit = solver.cfg.kappa_g * min(delta, ng)
            if ng + e <= tau or e <= limit * (1. + 1e-12):
                break
            if limit <= 0:
                raise RuntimeError('Zero certified forcing budget at nonterminating point')
            budget = min(.7 * limit, .25 * budget)
            p0 = pi
            self.pde.counts['gradient_refinements'] += 1
            solver.gradient_refinements += 1
        else:
            raise RuntimeError('Unable to certify dead-zone dynamic gradient forcing')

        self.previous_adjoint[idx] = pi
        solver.gradient_evals += 1
        solver.max_grad_error = max(solver.max_grad_error, e)
        solver.max_gradient_error = max(solver.max_gradient_error, e)
        if ng + e > tau and limit > 0 and np.isfinite(limit):
            ratio = e / limit
            solver.max_grad_ratio = max(solver.max_grad_ratio, ratio)
            solver.max_gradient_ratio = max(solver.max_gradient_ratio, ratio)

        value = smooth_value(m, z, y, solver.cfg.alpha)
        if not solver.cfg.smooth_only:
            value += dead_zone_moreau_value(m, z, mu, solver.cfg.beta, self.c)
        result = Evaluation(value, g, G, float(e), y, p, float(ry))
        solver.cost[ell]['eval_calls'] += 1
        solver.cost[ell]['eval_time'] += time.perf_counter() - t0
        return result

    def make_model(self, solver, ell, z, mu, ev=None):
        m = solver.levels[ell].mesh
        if solver.cfg.hessian_mode == 'exact':
            H = ExactReducedHessian(m, solver.cfg.alpha, stats=solver.hessian_stats)
        elif solver.cfg.hessian_mode == 'lbfgs':
            H = LBFGSHessian(m.W, solver.cfg.curvature, solver.lbfgs_pairs[ell])
        else:
            raise ValueError('DeadZone backend supports exact or lbfgs Hessian modes')
        if solver.cfg.smooth_only:
            return H
        D = dead_zone_moreau_hessian_diagonal(z, mu, solver.cfg.beta, self.c)
        return HessianPlusDiagonal(H, D)

    def original_stationarity(self, solver, z, r, eps_hint):
        """Certify the original *unconstrained* dead-zone prox residual."""
        ell = len(solver.levels) - 1
        m = solver.levels[ell].mesh
        idx = self.pde_index(solver, ell)
        y, ry = self.pde.state(idx, z)
        budget = max(1e-12, .025 * eps_hint)
        p, e, pi = self.pde.adjoint(
            idx, y, budget, ry, p0=self.previous_adjoint[idx]
        )
        self.previous_adjoint[idx] = pi
        g = smooth_gradient(m, z, p, solver.cfg.alpha)
        psi = norm(m, z - dead_zone_prox(z - r*g, r, solver.cfg.beta, self.c)) / r
        return float(psi + e), float(psi), float(e)
