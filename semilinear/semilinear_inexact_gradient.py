#Inexact gradient
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

#mesh generation


from dataclasses import dataclass


@dataclass
class TriMesh:
    nodes: np.ndarray       # shape (Nnodes, 2)
    elements: np.ndarray    # shape (Nelem, 3)
    boundary: np.ndarray    # boolean array, shape (Nnodes,)
    interior: np.ndarray    # indices of interior nodes


def create_unit_square_mesh(n):
    """
    Uniform triangular mesh of Omega=(0,1)^2.

    Each square is divided into two triangles.

    Number of triangles = 2*n^2.

    Hence:
        n=64  -> 8192 triangles
        n=128 -> 32768 triangles
        n=256 -> 131072 triangles
    """

    x = np.linspace(0.0, 1.0, n + 1)
    y = np.linspace(0.0, 1.0, n + 1)

    X, Y = np.meshgrid(x, y, indexing="ij")

    nodes = np.column_stack((X.ravel(), Y.ravel()))

    def idx(i, j):
        return i * (n + 1) + j

    elements = []

    for i in range(n):
        for j in range(n):

            n00 = idx(i, j)
            n10 = idx(i + 1, j)
            n01 = idx(i, j + 1)
            n11 = idx(i + 1, j + 1)

            # two counter-clockwise triangles
            elements.append([n00, n10, n11])
            elements.append([n00, n11, n01])

    elements = np.asarray(elements, dtype=int)

    tol = 1e-14
    boundary = (
        (np.abs(nodes[:, 0]) < tol)
        | (np.abs(nodes[:, 0] - 1.0) < tol)
        | (np.abs(nodes[:, 1]) < tol)
        | (np.abs(nodes[:, 1] - 1.0) < tol)
    )

    interior = np.where(~boundary)[0]

    return TriMesh(
        nodes=nodes,
        elements=elements,
        boundary=boundary,
        interior=interior,
    )

# P_1 stiffness matrix K

def element_geometry(mesh):
    """
    Compute element areas and gradients of P1 basis functions.

    Returns
    -------
    areas : (Nelem,)
    grads : (Nelem, 3, 2)
        grads[e,i,:] = grad(phi_i) on element e.
    """

    pts = mesh.nodes[mesh.elements]  # (Nelem,3,2)

    x0 = pts[:, 0, 0]
    y0 = pts[:, 0, 1]

    x1 = pts[:, 1, 0]
    y1 = pts[:, 1, 1]

    x2 = pts[:, 2, 0]
    y2 = pts[:, 2, 1]

    twice_area = (
        (x1 - x0) * (y2 - y0)
        - (x2 - x0) * (y1 - y0)
    )

    areas = 0.5 * np.abs(twice_area)

    grads = np.empty((len(areas), 3, 2))

    grads[:, 0, 0] = (y1 - y2) / twice_area
    grads[:, 0, 1] = (x2 - x1) / twice_area

    grads[:, 1, 0] = (y2 - y0) / twice_area
    grads[:, 1, 1] = (x0 - x2) / twice_area

    grads[:, 2, 0] = (y0 - y1) / twice_area
    grads[:, 2, 1] = (x1 - x0) / twice_area

    return areas, grads


def assemble_stiffness(mesh):
    """
    Assemble P1 stiffness matrix
        K_ij = int grad(phi_j) . grad(phi_i).
    """

    areas, grads = element_geometry(mesh)

    Ne = mesh.elements.shape[0]
    Nn = mesh.nodes.shape[0]

    # local stiffness matrices
    Kloc = (
        areas[:, None, None]
        * np.einsum("eik,ejk->eij", grads, grads)
    )

    rows = np.repeat(mesh.elements, 3, axis=1).ravel()
    cols = np.tile(mesh.elements, (1, 3)).ravel()

    data = Kloc.ravel()

    K = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(Nn, Nn)
    ).tocsr()

    return K

# P_0 control b for u

def assemble_control_load(mesh, z):
    """
    Assemble
        b_i = int_Omega z_h phi_i dx

    where z is P0: one value per triangular element.

    Parameters
    ----------
    z : array, shape (Nelem,)
    """

    areas, _ = element_geometry(mesh)

    Ne = mesh.elements.shape[0]
    Nn = mesh.nodes.shape[0]

    if len(z) != Ne:
        raise ValueError("z must contain one value per triangle.")

    bloc = areas[:, None] * z[:, None] / 3.0
    bloc = np.repeat(bloc, 3, axis=1)

    b = np.zeros(Nn)

    np.add.at(
        b,
        mesh.elements.ravel(),
        bloc.ravel()
    )

    return b

#Newton solve for u
def triangle_quadrature():
    """
    7-point Dunavant quadrature.
    Weights sum to 1, so physical integration is
        |T| * sum_q w_q f(x_q).
    """

    bary = []
    weights = []

    # centroid
    bary.append([1/3, 1/3, 1/3])
    weights.append(0.225)

    a = 0.059715871789770
    b = 0.470142064105115
    w = 0.132394152788506

    bary.extend([
        [a, b, b],
        [b, a, b],
        [b, b, a],
    ])
    weights.extend([w, w, w])

    a = 0.797426985353087
    b = 0.101286507323456
    w = 0.125939180544827

    bary.extend([
        [a, b, b],
        [b, a, b],
        [b, b, a],
    ])
    weights.extend([w, w, w])

    return np.asarray(bary), np.asarray(weights)

def assemble_nonlinear_terms(mesh, u):
    """
    Assemble

        N_i(u) = int u_h^3 phi_i dx

    and

        J_ij(u) = int 3 u_h^2 phi_i phi_j dx.
    """

    areas, _ = element_geometry(mesh)
    bary, weights = triangle_quadrature()

    elements = mesh.elements
    Ne = elements.shape[0]
    Nn = mesh.nodes.shape[0]

    # nodal values on each triangle
    ue = u[elements]                     # (Ne,3)

    # u at quadrature points
    uq = ue @ bary.T                     # (Ne,nq)

    # nonlinear residual
    Nloc = np.einsum(
        "eq,q,qi,e->ei",
        uq**3,
        weights,
        bary,
        areas
    )

    Nvec = np.zeros(Nn)

    np.add.at(
        Nvec,
        elements.ravel(),
        Nloc.ravel()
    )

    # nonlinear Jacobian
    Jloc = np.einsum(
        "eq,q,qi,qj,e->eij",
        3.0 * uq**2,
        weights,
        bary,
        bary,
        areas
    )

    rows = np.repeat(elements, 3, axis=1).ravel()
    cols = np.tile(elements, (1, 3)).ravel()

    J = sp.coo_matrix(
        (Jloc.ravel(), (rows, cols)),
        shape=(Nn, Nn)
    ).tocsr()

    return Nvec, J

def solve_state(
    mesh,
    z,
    tol=1e-8,
    max_newton=30,
    verbose=False,
    u0=None,
):
    """
    Solve
        -Delta u + u^3 = z
        u = 0 on boundary

    using P1 FEM + Newton.

    z is P0: one value per triangle.
    """

    K = assemble_stiffness(mesh)
    b = assemble_control_load(mesh, z)

    Nn = mesh.nodes.shape[0]
    I = mesh.interior

    if u0 is None:
        u = np.zeros(Nn)
    else:
        u = np.array(u0, dtype=float, copy=True)

    # enforce boundary values exactly
    u[mesh.boundary] = 0.0

    for it in range(max_newton):

        Nvec, Jnl = assemble_nonlinear_terms(mesh, u)

        residual = K @ u + Nvec - b
        rI = residual[I]

        resnorm = np.linalg.norm(rI)

        if verbose:
            print(
                f"Newton {it:2d}: "
                f"||r|| = {resnorm:.6e}"
            )

        if resnorm <= tol:
            return u

        A = K + Jnl

        AII = A[I][:, I]

        duI = spla.spsolve(AII, -rI)

        u[I] += duI
        u[mesh.boundary] = 0.0

    raise RuntimeError(
        f"State Newton solver did not converge "
        f"after {max_newton} iterations."
    )
    
# solve adjoint equation p
def assemble_p1_load(mesh, u, w_value=-1.0):
    """
    Assemble
        b_i = int (w-u_h) phi_i dx

    for constant desired state w.
    """

    areas, _ = element_geometry(mesh)
    bary, weights = triangle_quadrature()

    elements = mesh.elements
    Nn = mesh.nodes.shape[0]

    ue = u[elements]
    uq = ue @ bary.T

    fq = w_value - uq

    bloc = np.einsum(
        "eq,q,qi,e->ei",
        fq,
        weights,
        bary,
        areas
    )

    b = np.zeros(Nn)

    np.add.at(
        b,
        elements.ravel(),
        bloc.ravel()
    )

    return b

def solve_adjoint(
    mesh,
    u,
    w_value=-1.0,
    rtol=1e-8,
    maxiter=1000,
    verbose=False,
):
    """
    Solve
        -Delta p + 3 u^2 p = w-u,
        p = 0 on boundary.

    Uses conjugate gradients so that the adjoint solve
    can deliberately be made inexact.
    """

    K = assemble_stiffness(mesh)

    # We only need the Jacobian part here
    _, Jnl = assemble_nonlinear_terms(mesh, u)

    A = K + Jnl

    b = assemble_p1_load(
        mesh,
        u,
        w_value=w_value
    )

    I = mesh.interior

    AII = A[I][:, I]
    bI = b[I]

    pI, info = spla.cg(
        AII,
        bI,
        rtol=rtol,
        atol=0.0,
        maxiter=maxiter,
    )

    if info < 0:
        raise RuntimeError("CG failed in adjoint solve.")

    if verbose:
        residual = bI - AII @ pI

        print(
            "Adjoint residual:",
            np.linalg.norm(residual),
            "CG info:",
            info,
        )

    p = np.zeros(mesh.nodes.shape[0])
    p[I] = pI

    return p

#inexact gradient
def project_p1_to_p0(mesh, p):
    """
    L2 projection of a P1 function onto P0.

    For a linear function on a triangle,
    the cell average is the average of the
    three nodal values.
    """

    pe = p[mesh.elements]

    return np.mean(pe, axis=1)

def reduced_gradient(mesh, z, p, alpha=1e-4):
    """
    Compute the P0 reduced L2-gradient

        grad f(z) = alpha*z - Pi_0 p.
    """

    p0 = project_p1_to_p0(mesh, p)

    return alpha * z - p0

def p0_l2_norm(mesh, q):
    areas, _ = element_geometry(mesh)
    return np.sqrt(np.sum(areas * q**2))


def compute_inexact_gradient(
    mesh,
    z,
    alpha=1e-4,
    w_value=-1.0,
    state_tol=1e-6,
    adjoint_tol=1e-6,
    verbose=False,
):
    """
    State + adjoint + reduced gradient.
    """

    u = solve_state(
        mesh,
        z,
        tol=state_tol,
        verbose=verbose,
    )

    p = solve_adjoint(
        mesh,
        u,
        w_value=w_value,
        rtol=adjoint_tol,
        verbose=verbose,
    )

    g = reduced_gradient(
        mesh,
        z,
        p,
        alpha=alpha,
    )

    return u, p, g

# level A^0: 8192 control DOFs
mesh = create_unit_square_mesh(64)

Ne = mesh.elements.shape[0]

# initial control
z = np.zeros(Ne)

u, p, g = compute_inexact_gradient(
    mesh,
    z,
    alpha=1e-4,
    w_value=-1.0,
    state_tol=1e-6,
    adjoint_tol=1e-6,
    verbose=True,
)

print("number of control DOFs =", len(z))
print("||g||_2 =", p0_l2_norm(mesh, g))
