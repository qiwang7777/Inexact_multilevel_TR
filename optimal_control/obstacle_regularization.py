"""P1 nodal obstacle indicator in the lumped finite-element mass metric.

C_h = {u in R^N : u_i >= psi_i for every mesh vertex}.
phi_h = I_C, prox_{mu phi}=projection onto C (independent of mu).
"""
from __future__ import annotations

import numpy as np
from .obstacle_mesh import obstacle_profile, forcing_profile, diffusion_profile


def obstacle_prox(z, psi):
    """Weighted W-prox of indicator C: max(z,psi) componentwise."""
    return np.maximum(np.asarray(z, dtype=float), np.asarray(psi, dtype=float))


def obstacle_phi(z, psi, tol=0.0):
    """Indicator functional: finite if and only if all nodal constraints hold."""
    return 0.0 if np.all(np.asarray(z) >= np.asarray(psi)-tol) else np.inf


def obstacle_moreau_value(weights, z, psi, mu):
    if mu <= 0:
        raise ValueError('mu must be positive')
    d = np.minimum(np.asarray(z, dtype=float)-np.asarray(psi, dtype=float), 0.)
    return float(0.5 / mu * np.dot(np.asarray(weights, dtype=float), d*d))


def obstacle_moreau_grad(z, psi, mu):
    """W-Riesz gradient of M_mu(I_C)."""
    if mu <= 0:
        raise ValueError('mu must be positive')
    return np.minimum(np.asarray(z, dtype=float)-np.asarray(psi, dtype=float), 0.) / mu


def obstacle_moreau_hessian_diagonal(z, psi, mu):
    """Generalized W-Hessian: 1/mu on violated nodes, zero otherwise.

    At u_i=psi_i, zero is a valid generalized Hessian selection.
    """
    if mu <= 0:
        raise ValueError('mu must be positive')
    return np.where(np.asarray(z) < np.asarray(psi), 1.0/mu, 0.)


def obstacle_continuation_warm_start(z, psi, mu_old, mu_new):
    """Preserve the Moreau gradient when reducing mu (before smooth change)."""
    if not (0 < mu_new < mu_old):
        raise ValueError('Require 0 < mu_new < mu_old')
    z = np.asarray(z, dtype=float)
    v = obstacle_prox(z, psi)
    return v + (mu_new/mu_old)*(z-v)
