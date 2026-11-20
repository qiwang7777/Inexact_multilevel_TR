"""P1 elliptic control backend for the ONE IRMTR algorithm in irmtr.py.

This class contains no trust-region steps, recursive IRMTR, or continuation
logic.  It supplies objective, certified gradient, exact reduced Hessian,
control-space metric, and proximal stationarity to IRMTRMoreau.
"""
import time
import numpy as np

from ..types import Evaluation
from ..regularization import (prox_phi, moreau_value, moreau_grad,
                              moreau_hessian_diagonal)
from ..hessian_models import LBFGSHessian, HessianPlusDiagonal
from .elliptic import (inner, norm, smooth_value, smooth_gradient,
                       ExactReducedHessian)


class EllipticBackend:
    control_space = 'p1'

    def __init__(self, pde):
        self.pde = pde
        self.index = {m.n: i for i,m in enumerate(pde.meshes)}
        self.cached_states = {}
        self.previous_adjoint = [None for _ in pde.meshes]

    def pde_index(self, solver, ell):
        return self.index[solver.levels[ell].mesh.n]

    @staticmethod
    def inner(mesh, x, y):
        return inner(mesh, x, y)

    @staticmethod
    def norm(mesh, x):
        return norm(mesh, x)

    def objective(self, solver, ell, z, mu,
                  return_state=False, u_initial=None):
        t0 = time.perf_counter()
        if not np.all(np.isfinite(z)):
            value, state = np.inf, None
        else:
            m = solver.levels[ell].mesh
            y, ry = self.pde.state(self.pde_index(solver,ell), z)
            self.cached_states[ell] = (z.copy(), y.copy(), ry)
            value = smooth_value(m,z,y,solver.cfg.alpha)
            if not solver.cfg.smooth_only:
                value += moreau_value(m,z,mu,solver.cfg.beta)
            state = (y, ry)
        solver.cost[ell]['obj_calls'] += 1
        solver.cost[ell]['obj_time'] += time.perf_counter()-t0
        return (value, state) if return_state else value

    def evaluate(self, solver, ell, z, mu, tau, state=None,
                 u_initial=None, p_initial=None, delta=None):
        t0 = time.perf_counter()
        m = solver.levels[ell].mesh
        if state is not None:
            # The generic core passes back the accepted objective's state
            # or the cached evaluation's (u,J).  For this linear backend,
            # J stores the certified state residual norm, not a Jacobian.
            y,ry = state
            solver.cost[ell]['reused_trial_states'] += 1
            self.cached_states.pop(ell, None)
        else:
            cached = self.cached_states.pop(ell, None)
            if cached is not None and np.array_equal(cached[0], z):
                y, ry = cached[1], cached[2]
                solver.cache_hits += 1
                solver.cost[ell]['reused_trial_states'] += 1
            else:
                y, ry = self.pde.state(self.pde_index(solver,ell), z)
        if delta is None:
            delta = np.inf
        # Certified forcing: the original linear PDE estimator supplies a
        # bound on ||approximate g - discrete exact g|| in the W norm.
        budget = solver.cfg.kappa_g*min(delta, max(tau,1e-14))
        idx = self.pde_index(solver,ell)
        p0 = self.previous_adjoint[idx]
        # If the core passes a full P1 p, use its interior components.
        if p_initial is not None:
            p0 = np.asarray(p_initial)[m.interior]
        for attempt in range(12):
            p, e, pi = self.pde.adjoint(idx,y,budget,ry,p0=p0)
            g = smooth_gradient(m,z,p,solver.cfg.alpha)
            G = g if solver.cfg.smooth_only else g + moreau_grad(z,mu,solver.cfg.beta)
            ng = norm(m,G)
            limit = solver.cfg.kappa_g*min(delta,ng)
            if ng+e <= tau or e <= limit*(1+1e-12):
                break
            if limit <= 0:
                raise RuntimeError('Zero dynamic gradient forcing at a nonterminating point')
            budget = min(.7*limit,.25*budget)
            p0 = pi
            self.pde.counts['gradient_refinements'] += 1
            solver.gradient_refinements += 1
        else:
            raise RuntimeError('Unable to certify elliptic dynamic gradient forcing')
        self.previous_adjoint[idx] = pi
        solver.gradient_evals += 1
        solver.max_grad_error = max(solver.max_grad_error,e)
        solver.max_gradient_error = max(solver.max_gradient_error,e)
        if ng+e > tau and limit > 0 and np.isfinite(limit):
            ratio = e/limit
            solver.max_grad_ratio = max(solver.max_grad_ratio,ratio)
            solver.max_gradient_ratio = max(solver.max_gradient_ratio,ratio)
        value = smooth_value(m,z,y,solver.cfg.alpha)
        if not solver.cfg.smooth_only:
            value += moreau_value(m,z,mu,solver.cfg.beta)
        result = Evaluation(value,g,G,float(e),y,p,float(ry))
        solver.cost[ell]['eval_calls'] += 1
        solver.cost[ell]['eval_time'] += time.perf_counter()-t0
        return result

    def make_model(self, solver, ell, z, mu, ev=None):
        m = solver.levels[ell].mesh
        if solver.cfg.hessian_mode == 'exact':
            H = ExactReducedHessian(m,solver.cfg.alpha,stats=solver.hessian_stats)
        elif solver.cfg.hessian_mode == 'lbfgs':
            H = LBFGSHessian(m.W,solver.cfg.curvature,solver.lbfgs_pairs[ell])
        else:
            raise ValueError('Elliptic backend supports exact or lbfgs Hessians')
        if solver.cfg.smooth_only:
            return H
        D = moreau_hessian_diagonal(z,mu,solver.cfg.beta)
        return HessianPlusDiagonal(H,D)

    def original_stationarity(self, solver, z, r, eps_hint):
        ell = len(solver.levels)-1
        m = solver.levels[ell].mesh
        idx = self.pde_index(solver,ell)
        y, ry = self.pde.state(idx,z)
        budget = max(1e-12,.025*eps_hint)
        p,e,pi = self.pde.adjoint(idx,y,budget,ry,
                                  p0=self.previous_adjoint[idx])
        self.previous_adjoint[idx] = pi
        g = smooth_gradient(m,z,p,solver.cfg.alpha)
        psi = norm(m,z-prox_phi(z-r*g,r,solver.cfg.beta))/r
        return float(psi+e), float(psi), float(e)
