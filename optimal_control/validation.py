"""Finite-difference checks for Hessian and Moreau curvature."""

import numpy as np
from .geometry import p0_inner
from .regularization import moreau_grad, moreau_hessian_diagonal
from .irmtr import IRMTRMoreau


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


