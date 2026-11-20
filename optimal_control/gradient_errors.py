"""Residual-based, computable gradient error certificates in FE geometry.

A certificate is for the EXACTLY DISCRETIZED state/adjoint equations.
It does not bound spatial discretization error or roundoff error.
"""
import numpy as np
from .geometry import element_geometry, get_stiffness, assemble_control_load, triangle_quadrature

# Poincare on (0,1)^2: 2*pi**2 ||v||_L2**2 <= ||grad v||_L2**2.
# Consistent P1 mass M >= W/4, so K >= (pi**2/2) W (on H01 interior).
# Applies to the discrete cubic monotone state operator and Jacobian.
SEMILINEAR_STABILITY_C = 2.0 / np.pi**2


def lumped_p1_weights(mesh):
    """P1 mass-lumped weights at ALL vertices; cache on mesh."""
    if not hasattr(mesh, '_cert_lumped_mass'):
        areas, _ = element_geometry(mesh)
        mesh._cert_lumped_mass = np.bincount(
            mesh.elements.ravel(),
            weights=np.repeat(areas / 3.0, 3),
            minlength=len(mesh.nodes))
    return mesh._cert_lumped_mass


def semilinear_gradient_certificate(mesh, z, u, p, w_value, J=None):
    """Bound ||g_tilde - g_exact_discrete||_{L2(P0)} from PDE residuals.

    State: K u + N(u) = B z (on interior nodes), N_i=(u^3,phi_i).
    Adjoint: (K+J(u))p = (w-u,phi_i), J_ij=(3u^2 phi_j,phi_i).

    Let C=2/pi^2 and ||v||_W^2=sum_i W_i v_i^2 for P1 nodal
    lumped mass. Monotonicity and the discrete Poincare inequality give
       ||u-u*||_W <= C ||r_state||_{W_I^-1} = E_u.
    Using the adjoint equations at u and u*, with P1 sup-norm stability:
       ||p-p*||_W <= C [||r_adj||_{W_I^-1}
           + (1 + 3(2||u||_infty + E_u/sqrt(min W_I))||p||_infty) E_u].
    Since L2 projection onto P0 is contractive and ||v||_L2<=||v||_W,
    the same bound holds for the P0 reduced-gradient error.

    These are conservative bounds for the finite-element problem only.
    """
    from .pde.semilinear import assemble_nonlinear_terms, assemble_p1_load
    w = lumped_p1_weights(mesh)
    I = mesh.interior
    if len(I) == 0:
        return 0.0, 0.0, 0.0
    K = get_stiffness(mesh)
    #nonlinear, jacobian = assemble_nonlinear_terms(mesh, u)
    if J is None:
        nonlinear, J = assemble_nonlinear_terms(mesh, u)
    else:
        areas,_ = element_geometry(mesh)
        bary, weights = triangle_quadrature()
        E = mesh.elements
        uq = u[E] @ bary.T
        nonlinear_local = np.einsum('eq,q,qi,e->ei',uq**3, weights, bary, areas)
        nonlinear = np.bincount(E.ravel(), weights=nonlinear_local.ravel(),minlength=len(mesh.nodes))
    ru = (K@u + nonlinear - assemble_control_load(mesh, z))[I]
    rp = ((K+J)@p - assemble_p1_load(mesh, u, w_value))[I]
    dual = lambda r: float(np.sqrt(np.dot(r/w[I], r)))
    eu = SEMILINEAR_STABILITY_C * dual(ru)
    linf_u = float(np.max(np.abs(u)))
    linf_p = float(np.max(np.abs(p)))
    # ||u-u*||_infty <= ||u-u*||_W/sqrt(min W_I), since P1 sup norm
    # is attained at a vertex and the boundary difference vanishes.
    eu_infty = eu / np.sqrt(float(np.min(w[I])))
    coefficient = 1.0 + 3.0 * (2.0*linf_u + eu_infty) * linf_p
    ep = SEMILINEAR_STABILITY_C * (dual(rp) + coefficient*eu)
    if not np.isfinite(ep):
        raise RuntimeError('Nonfinite certified gradient error')
    return float(ep), float(eu), float(dual(rp))
