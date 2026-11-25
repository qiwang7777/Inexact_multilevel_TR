"""Diagnostics for a discrete quadratic obstacle problem with W-metric MY.

The smooth objective is f(z)=0.5*z.T@A@z-b.T@z, with
W=diag(weights)>0, K={z>=psi}, and a mass-lumped MY envelope.

For the P1 Neumann obstacle problem in this example, A=stiffness+M,
and A >= (1/4) W in the quadratic-form sense (triangular P1 M).
This gives a primal-dual *objective gap* certificate, distinct from a
proximal gradient residual certificate.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Metrics:
    my_stationarity: float
    raw_prox_residual: float
    feasible_prox_residual: float
    raw_prox_bound: float
    infeasibility: float
    complementarity: float
    F_mu: float
    feasible_objective: float
    objective_gap_upper: float

    def asdict(self):
        return asdict(self)


def obstacle_metrics(A, b, weights, psi, z, mu: float, r: float) -> Metrics:
    """Calculate diagnostics with exact smooth gradients, no PDE error term.

    Raw PG:   ||z - P_K(z-r*grad_W f(z))||_W / r
    Feas PG:  ||y - P_K(y-r*grad_W f(y))||_W / r, y=P_K(z)
    
    Raw PG <= ||grad_W F_mu(z)||_W + ||z-y||_W/r,
    for the componentwise LOWER obstacle projection (the exact identity
    P_K(z-r*lambda)=P_K(z), lambda=(P_K(z)-z)/mu >= 0).

    Objective gap bound uses A >= 0.25 W, specific to this FEM example.
    """
    z=np.asarray(z,dtype=float).reshape(-1)
    b=np.asarray(b,dtype=float).reshape(-1)
    weights=np.asarray(weights,dtype=float).reshape(-1)
    psi=np.asarray(psi,dtype=float).reshape(-1)
    if not (mu>0 and r>0 and np.all(weights>0)):
        raise ValueError('Require mu,r,all weights > 0')
    if any(a.size!=z.size for a in (b,weights,psi)):
        raise ValueError('Incompatible vector lengths')
    y=np.maximum(z,psi)
    def wnorm(x):
        return float(np.sqrt(max(0.,np.dot(weights*x,x))))

    Az=np.asarray(A@z).reshape(-1)
    Ay=np.asarray(A@y).reshape(-1)
    grad_z=(Az-b)/weights
    grad_y=(Ay-b)/weights
    nu=(y-z)/mu
    inner=wnorm(grad_z-nu)
    raw_pg=wnorm((z-np.maximum(z-r*grad_z,psi))/r)
    feas_pg=wnorm((y-np.maximum(y-r*grad_y,psi))/r)
    d=wnorm(y-z)
    raw_bound=inner+d/r
    if raw_pg>raw_bound+max(1e-10,1e-9*raw_bound):
        raise AssertionError('Obstacle raw PG theoretical upper bound failed')

    smooth_z=float(.5*np.dot(z,Az)-np.dot(b,z))
    fmu=smooth_z + d*d/(2*mu)
    feasible_objective=float(.5*np.dot(y,Ay)-np.dot(b,y))
    # L(z,lambda)=f(z)-<lambda,z-psi>_W = F_mu(z)+d^2/(2*mu)
    # gap <= f(y)-L(z,lambda)+(1/(2*m))*||grad L(z)||_W^2, m=1/4
    gap=max(0.,feasible_objective-fmu-d*d/(2*mu)+2*inner**2)
    return Metrics(inner,raw_pg,feas_pg,raw_bound,d,d*d/mu,
                   fmu,feasible_objective,gap)
