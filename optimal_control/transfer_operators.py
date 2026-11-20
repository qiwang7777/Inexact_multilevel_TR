"""Control-space transfers for the two PDE-constrained optimization examples.

The optimization restriction R is the adjoint of prolongation P with respect
to the chosen discrete control-space inner products:

    <P c, f>_fine = <c, R f>_coarse.

For P0 (uniform triangle areas), R is four-child averaging and P = 4 R.T.
For P1 (lumped nodal mass W), P is interpolation and R = Wc^-1 P.T Wf.

The PDE multigrid operator in the elliptic example uses the *same P1
prolongation*, but its residual restriction is P.T, not optimization R.
"""

import numpy as np
import scipy.sparse as sp


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


def p1_transfers(coarse, fine):
    """R and P; coarse diagonals always lower-left to upper-right."""
    nc,nf=coarse.n,fine.n
    if nf!=2*nc:raise ValueError('Adjacent meshes must be 2:1 refinements')
    idx=lambda i,j:i*(nc+1)+j
    rows=[];cols=[];vals=[]
    for row,(x,y) in enumerate(fine.nodes):
        i=min(int(np.floor(nc*x+1e-12)),nc-1)
        j=min(int(np.floor(nc*y+1e-12)),nc-1)
        xi,eta=nc*x-i,nc*y-j
        if eta<=xi:
            vertices=(idx(i,j),idx(i+1,j),idx(i+1,j+1))
            coeffs=(1-xi,xi-eta,eta)
        else:
            vertices=(idx(i,j),idx(i+1,j+1),idx(i,j+1))
            coeffs=(1-eta,xi,eta-xi)
        for v,q in zip(vertices,coeffs):
            if q>1e-14:
                rows.append(row);cols.append(v);vals.append(q)
    P=sp.coo_matrix((vals,(rows,cols)),shape=(len(fine.nodes),len(coarse.nodes))).tocsr()
    R=(sp.diags(1/coarse.W)@P.T@sp.diags(fine.W)).tocsr()
    if not np.allclose(P@np.ones(P.shape[1]),1.,atol=1e-12):
        raise AssertionError('P must preserve constants')
    if not np.allclose(R@np.ones(R.shape[1]),1.,atol=1e-12):
        raise AssertionError('R must preserve constants')
    return R,P


def build_transfer_operators(coarse, fine, control_space):
    """Return (R, P) for control-space type 'p0' or 'p1'."""
    name = str(control_space).strip().lower()
    if name == 'p0':
        return p0_transfers(coarse, fine)
    if name == 'p1':
        return p1_transfers(coarse, fine)
    raise ValueError(f"Unsupported control space {control_space!r}; use 'p0' or 'p1'")
