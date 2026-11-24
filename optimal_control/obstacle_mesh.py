"""Full P1 FEM meshes for the Neumann obstacle variational inequality.

The coefficient, load, and obstacle are those stated in the manuscript:
  a(x1,x2) = 1 + 4 sin(2*pi*x1)^2 sin(2*pi*x2)^2
  g(x1,x2) = 10 cos(pi*x1) cos(pi*x2)
  psi(x1,x2) = .1 exp(-40*((x1-.5)^2+(x2-.5)^2))-.05

The 7-point triangle rule integrates the non-polynomial a and g accurately.
All P1 nodes, including boundary nodes, remain optimization DOF: the natural
boundary condition is homogeneous Neumann, NOT homogeneous Dirichlet.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

from .geometry import create_unit_square_mesh, element_geometry, triangle_quadrature


def diffusion_profile(x):
    """Positive oscillatory coefficient, 1 <= a(x) <= 5."""
    x = np.asarray(x, dtype=float)
    return 1.0 + 4.0 * np.sin(2*np.pi*x[..., 0])**2 * np.sin(2*np.pi*x[..., 1])**2


def forcing_profile(x):
    x = np.asarray(x, dtype=float)
    return 10.0 * np.cos(np.pi*x[..., 0]) * np.cos(np.pi*x[..., 1])


def obstacle_profile(x):
    x = np.asarray(x, dtype=float)
    r2 = (x[..., 0]-.5)**2 + (x[..., 1]-.5)**2
    return .1*np.exp(-40.*r2)-.05


@dataclass
class ObstacleMesh:
    n: int
    nodes: np.ndarray
    triangles: np.ndarray
    interior: np.ndarray
    M: sp.csr_matrix
    W: np.ndarray
    A: sp.csr_matrix
    load: np.ndarray
    psi: np.ndarray
    a_nodes: np.ndarray


def mesh_level(n: int) -> ObstacleMesh:
    """Assemble A=K_a+M and b_i=int g*phi_i on the full P1 nodal space."""
    if int(n) != n or n < 2:
        raise ValueError('n must be an integer >= 2')
    n = int(n)
    base = create_unit_square_mesh(n)
    x = base.nodes
    E = base.elements
    pts = x[E]
    area, grad = element_geometry(base)
    bary, qw = triangle_quadrature()
    xq = np.einsum('eic,qi->eqc', pts, bary)
    a_mean = diffusion_profile(xq) @ qw

    stiffness = (area*a_mean)[:,None,None]*np.einsum(
        'eic,ejc->eij', grad, grad)
    mass = (area/12.)[:,None,None]*np.array(
        [[2.,1.,1.],[1.,2.,1.],[1.,1.,2.]])

    rows = np.repeat(E, 3, axis=1).ravel()
    cols = np.tile(E, (1,3)).ravel()
    shape = (len(x),len(x))
    M = sp.coo_matrix((mass.ravel(), (rows,cols)),shape=shape).tocsr()
    A = sp.coo_matrix(((stiffness+mass).ravel(), (rows,cols)),shape=shape).tocsr()
    W = np.asarray(M.sum(axis=1)).ravel()

    # Assemble true FE load int g(x)*phi_i(x) dx, not M@g(nodes).
    gq = forcing_profile(xq)
    load_local = area[:,None]*np.einsum('eq,q,qi->ei',gq,qw,bary)
    load = np.bincount(E.ravel(), weights=load_local.ravel(), minlength=len(x))

    return ObstacleMesh(n,x,E,base.interior,M,W,A,load,
                        obstacle_profile(x),diffusion_profile(x))
