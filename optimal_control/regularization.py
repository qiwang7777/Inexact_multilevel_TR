"""Box + L1 proximal map, Moreau envelope, gradients and generalized curvature."""

import numpy as np
from .geometry import element_geometry


def prox_phi(z, r, beta=1e-2):
    """L2 prox of beta||.||_1 + indicator_{[-25,25]}, r>0."""
    if r <= 0:
        raise ValueError('prox parameter must be positive')
    return np.clip(np.sign(z) * np.maximum(np.abs(z)-r*beta, 0.), -25., 25.)


def moreau_value(mesh, z, mu, beta=1e-2):
    v = prox_phi(z, mu, beta)
    # P0 controls use triangle areas; P1 controls use lumped nodal weights.
    weights = mesh.W if hasattr(mesh, "W") else element_geometry(mesh)[0]
    return float(np.dot(weights, beta*np.abs(v) + 0.5/mu*(z-v)**2))


def moreau_grad(z, mu, beta=1e-2):
    return (z-prox_phi(z,mu,beta))/mu


def moreau_hessian_diagonal(z, mu, beta):
    """A generalized Hessian selection for MY(beta*|z|+I_box).

    At switching points the envelope is C1 but not necessarily C2;
    choose 1/mu as an admissible endpoint value.  For beta=0,
    z=0 is strictly inside the box and has ZERO Moreau curvature.
    """
    if mu <= 0 or beta < 0:
        raise ValueError('mu must be positive and beta nonnegative')
    a = np.abs(z)
    soft = (beta > 0.0) & (a <= mu*beta)
    box = a >= 25.0 + mu*beta
    return (soft | box).astype(float)/mu


