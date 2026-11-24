"""Mass-lumped P1 dead-zone penalty and its Moreau--Yosida envelope.

Original discrete functional: phi(z) = beta * sum_i W_i max(|z_i|-c, 0).
The prox and envelope use the W-weighted Euclidean/Hilbert norm, so each
node is independent. W is *not* inserted in the scalar prox threshold.
"""
from __future__ import annotations

import numpy as np


def dead_zone_prox(v, t, beta=1e-2, c=2.0):
    """prox_{t*phi}(v) for scalar hinge phi(s)=beta*(|s|-c)_+.

    Works elementwise for ndarray inputs; t is a positive scalar.
    """
    if t <= 0 or beta < 0 or c < 0:
        raise ValueError('Require t>0, beta>=0, c>=0')
    v = np.asarray(v, dtype=float)
    av = np.abs(v)
    prox_abs = np.where(av <= c, av, c + np.maximum(av-c-t*beta, 0.))
    return np.copysign(prox_abs, v)


def dead_zone_phi(mesh, z, beta=1e-2, c=2.0):
    """Original mass-lumped nonsmooth functional (not the MY envelope)."""
    return float(beta * np.dot(mesh.W, np.maximum(np.abs(z)-c, 0.)))


def dead_zone_moreau_value(mesh, z, mu, beta=1e-2, c=2.0):
    """Moreau envelope value in the lumped-mass control metric."""
    if mu <= 0:
        raise ValueError('mu must be positive')
    d = np.maximum(np.abs(z) - c, 0.)
    # Huberisation of beta*d. Quadratic on [0,mu*beta], linear outside.
    point = np.where(d <= mu*beta, 0.5*d*d/mu, beta*d-0.5*mu*beta*beta)
    return float(np.dot(mesh.W, point))


def dead_zone_moreau_grad(z, mu, beta=1e-2, c=2.0):
    """W-Riesz gradient of MY envelope (not the Euclidean gradient)."""
    if mu <= 0:
        raise ValueError('mu must be positive')
    return (np.asarray(z) - dead_zone_prox(z, mu, beta, c)) / mu


def dead_zone_moreau_hessian_diagonal(z, mu, beta=1e-2, c=2.0):
    """A positive semidefinite generalized Hessian diagonal.

    Use 1/mu in the strictly quadratic shoulder, 0 outside;
    At the inner dead-zone boundary choose the zero-curvature side.
    """
    if mu <= 0:
        raise ValueError('mu must be positive')
    d = np.abs(z) - c
    return np.where((d > 0.) & (d <= mu*beta) & (beta > 0), 1./mu, 0.)
