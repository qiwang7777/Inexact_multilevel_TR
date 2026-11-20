"""P1 finite-element mesh, P0 controls, transfers, quadrature, and L2 geometry."""

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


@dataclass
class TriMesh:
    nodes: np.ndarray
    elements: np.ndarray
    boundary: np.ndarray
    interior: np.ndarray


def create_unit_square_mesh(n):
    """P1 triangle nodes; P0 control: 2*n*n elements."""
    x = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    nodes = np.column_stack((X.ravel(), Y.ravel()))
    idx = lambda i, j: i * (n + 1) + j
    elements = np.array(
        [tri for i in range(n) for j in range(n)
         for tri in ([idx(i,j), idx(i+1,j), idx(i+1,j+1)],
                     [idx(i,j), idx(i+1,j+1), idx(i,j+1)])],
        dtype=int,
    )
    boundary = ((nodes[:, 0] == 0) | (nodes[:, 0] == 1)
                | (nodes[:, 1] == 0) | (nodes[:, 1] == 1))
    return TriMesh(nodes, elements, boundary, np.flatnonzero(~boundary))


def element_geometry(mesh):
    if hasattr(mesh,'_geometry_cache'):
        return mesh._geometry_cache
    pts = mesh.nodes[mesh.elements]
    a, b, c = pts[:, 0], pts[:, 1], pts[:, 2]
    det = ((b[:,0]-a[:,0])*(c[:,1]-a[:,1])
           -(c[:,0]-a[:,0])*(b[:,1]-a[:,1]))
    grads = np.stack((
        np.column_stack((b[:,1]-c[:,1], c[:,0]-b[:,0])),
        np.column_stack((c[:,1]-a[:,1], a[:,0]-c[:,0])),
        np.column_stack((a[:,1]-b[:,1], b[:,0]-a[:,0])),
    ), axis=1) / det[:,None,None]
    mesh._geometry_cache = (np.abs(det)/2.0,grads)
    return mesh._geometry_cache


def triangle_quadrature():
    bary = np.array([
        [1/3, 1/3, 1/3],
        [.059715871789770, .470142064105115, .470142064105115],
        [.470142064105115, .059715871789770, .470142064105115],
        [.470142064105115, .470142064105115, .059715871789770],
        [.797426985353087, .101286507323456, .101286507323456],
        [.101286507323456, .797426985353087, .101286507323456],
        [.101286507323456, .101286507323456, .797426985353087],
    ])
    weights = np.array([.225] + [.132394152788506]*3
                             + [.125939180544827]*3)
    return bary, weights


def assemble_stiffness(mesh):
    areas, grads = element_geometry(mesh)
    E = mesh.elements
    Kloc = areas[:,None,None] * np.einsum('eik,ejk->eij',grads,grads)
    return sp.coo_matrix((Kloc.ravel(),
                          (np.repeat(E,3,axis=1).ravel(),
                           np.tile(E,(1,3)).ravel())),
                         shape=(len(mesh.nodes),)*2).tocsr()


def get_stiffness(mesh):
    if not hasattr(mesh, '_K_cached'):
        mesh._K_cached = assemble_stiffness(mesh)
    return mesh._K_cached


def assemble_control_load(mesh,z):
    areas, _ = element_geometry(mesh)
    if len(z)!=len(areas):
        raise ValueError('P0 control must have one coefficient per triangle')
    return np.bincount(mesh.elements.ravel(),
                       weights=np.repeat(areas*z/3.0,3),
                       minlength=len(mesh.nodes))


def p0_l2_norm(mesh,z):
    areas,_=element_geometry(mesh)
    return float(np.sqrt(np.dot(areas,z*z)))


def p0_inner(mesh, a, b):
    areas, _ = element_geometry(mesh)
    return float(np.dot(areas * a, b))


def p0_norm(mesh, a):
    return np.sqrt(max(0.0, p0_inner(mesh, a, a)))


def p0_transfers(mesh_coarse, mesh_fine):
    """L2 restriction=cell averaging and prolongation=R* for these meshes.

    Requires fine grid to have precisely twice the square divisions of coarse,
    and the diagonal convention in create_unit_square_mesh.
    """
    nc = int(round(np.sqrt(mesh_coarse.elements.shape[0]/2)))
    nf = int(round(np.sqrt(mesh_fine.elements.shape[0]/2)))
    if nf != 2*nc:
        raise ValueError('fine grid must be a single uniform refinement')
    cent = mesh_fine.nodes[mesh_fine.elements].mean(axis=1)
    scaled = cent*nc
    cell = np.minimum(np.floor(scaled).astype(int), nc-1)
    i,j=cell[:,0],cell[:,1]
    triangle = ((scaled[:,1]-j) > (scaled[:,0]-i)).astype(int)
    parent = 2*(i*nc+j)+triangle
    Nf=len(parent)
    Nc=mesh_coarse.elements.shape[0]
    R=sp.coo_matrix((np.full(Nf,.25),(parent,np.arange(Nf))),shape=(Nc,Nf)).tocsr()
    P=(4*R.T).tocsr()  # L2-adjoint, NOT ordinary Euclidean transpose
    if not (np.allclose(np.asarray(R.sum(axis=1)).ravel(),1.)
            and np.allclose(np.asarray(P.sum(axis=1)).ravel(),1.)):
        raise AssertionError('transfer not constant-preserving')
    return R,P


