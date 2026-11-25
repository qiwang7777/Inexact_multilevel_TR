#!/usr/bin/env python3
"""Independent primal-dual active-set / semismooth Newton refinement for
nodal P1 obstacle QP  min 0.5*u.T@A@u - b.T@u, u>=psi.

Run from the Inexact_multilevel_TR repository root after the obstacle driver:
  python obstacle_vi_refinement.py --input obstacle_irmtr_results/obstacle_single_solution.npz

This is a SEPARATE postprocessor, NOT part of the original IRMTR algorithm.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import spsolve


def pg_data(A, b, W, psi, u, r):
    g = np.asarray(A @ u - b, dtype=float).ravel() / W
    t = u - r*g
    F = u - np.maximum(t, psi)
    merit = 0.5 * float(np.dot(W, F*F))
    residual = (2.*merit)**0.5 / r
    active = (t <= psi)
    return residual, merit, F, active


def refine_obstacle(A, b, W, psi, initial, *, r=.05, eps=1e-4,
                    max_iter=40, max_backtracks=25, verbose=True):
    """Globalized semismooth Newton on R_r(u)=u-P_C(u-r*W^-1(Au-b)).

    Each generalized Newton step fixes active u_A=psi_A and solves the
    free-node elliptic system A_FF u_F=b_F-A_FA psi_A. An Armijo line
    search on 0.5 ||R_r||_W^2 globalization; fallback to merit gradient.
    The final certificate is ALWAYS recomputed at a feasible projection.
    """
    u = np.maximum(np.asarray(initial, dtype=float).copy(), psi)
    A = A.tocsr()
    b = np.asarray(b, dtype=float)
    W = np.asarray(W, dtype=float)
    psi = np.asarray(psi, dtype=float)
    history=[]
    for k in range(max_iter+1):
        res, merit, F, active = pg_data(A,b,W,psi,u,r)
        feasible = np.maximum(u,psi)
        feas_res,_,_,_ = pg_data(A,b,W,psi,feasible,r)
        violation = float(np.sqrt(np.dot(W,np.minimum(u-psi,0.)**2)))
        history.append(dict(k=k,residual=res,feasible_residual=feas_res,
                            infeasibility=violation,active_nodes=int(active.sum())))
        if verbose:
            print(f'PDAS k={k:02d} PG={res:.6e} PG_feas={feas_res:.6e} '
                  f'violation={violation:.3e} active={active.sum()}')
        if feas_res <= eps:
            return feasible, history, True
        if k==max_iter:
            break
        free = ~active
        trial = psi.copy()
        if np.any(free):
            rhs = b[free] - A[free,:][:,active] @ psi[active]
            trial[free] = spsolve(A[free,:][:,free].tocsc(),rhs)
        direction=trial-u
        if not np.all(np.isfinite(direction)):
            raise FloatingPointError('Non-finite PDAS direction')
        # Use the globalization merit, not the original energy at infeasible trials.
        alpha=1.
        accepted=False
        for _ in range(max_backtracks):
            candidate = u + alpha*direction
            _, cand_merit, _, _ = pg_data(A,b,W,psi,candidate,r)
            if cand_merit <= merit*(1.-1e-4*alpha):
                u=candidate
                accepted=True
                break
            alpha*=.5
        if accepted:
            continue
        # If a generalized Newton step is not a merit descent direction,
        # use W-steepest descent on the selected piecewise-smooth merit.
        # Generalized Jacobian rows are identity on active nodes and
        # r*diag(1/W)@A on free nodes.
        grad_merit = np.where(active,W*F,0.) + r*(A @ np.where(free,F,0.))
        direction=-grad_merit/W
        slope=float(np.dot(grad_merit,direction))
        alpha=min(1.,1./max(1.,float(np.max(np.asarray(abs(A).sum(axis=1)).ravel()/W))))
        for _ in range(max_backtracks+20):
            candidate=u+alpha*direction
            _,cand_merit,_,_=pg_data(A,b,W,psi,candidate,r)
            if cand_merit <= merit+1e-4*alpha*slope:
                u=candidate
                accepted=True
                break
            alpha*=.5
        if not accepted:
            if verbose:
                print('PDAS stopped: neither Newton nor fallback merit step succeeded')
            break
    feasible=np.maximum(u,psi)
    return feasible,history,False


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True,
                   help='Obstacle driver obstacle_single_solution.npz or obstacle_multi_solution.npz')
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--eps',type=float,default=1e-4)
    p.add_argument('--prox-r',type=float,default=.05)
    p.add_argument('--max-iter',type=int,default=40)
    args=p.parse_args()
    with np.load(args.input) as data:
        u0=np.asarray(data['z'],dtype=float).copy()
        nodes=np.asarray(data['nodes'],dtype=float)
        psi_saved=np.asarray(data['psi'],dtype=float)
    n=round(np.sqrt(len(nodes)))-1
    if (n+1)**2 != len(nodes):
        raise ValueError('Cannot infer a square P1 mesh dimension from node count')
    print(f'Reassembling original nodal P1 VI on mesh n={n}, DOF={len(nodes)}')
    from optimal_control.obstacle_mesh import mesh_level
    mesh=mesh_level(n)
    if not np.allclose(mesh.nodes,nodes,atol=1e-13,rtol=0):
        raise ValueError('Node order differs from the saved obstacle solution')
    if not np.allclose(mesh.psi,psi_saved,atol=1e-13,rtol=0):
        raise ValueError('Obstacle profile differs from saved solution')
    u,hist,ok=refine_obstacle(mesh.A,mesh.load,mesh.W,mesh.psi,u0,
                               r=args.prox_r,eps=args.eps,max_iter=args.max_iter)
    residual,_,_,_=pg_data(mesh.A,mesh.load,mesh.W,mesh.psi,u,args.prox_r)
    value=0.5*float(u@(mesh.A@u))-float(mesh.load@u)
    output=args.output or args.input.with_name(args.input.stem+'_pdas.npz')
    np.savez_compressed(output,z=u,PG=residual,objective=value,converged=ok,
                        iterations=len(hist)-1,nodes=nodes,psi=mesh.psi)
    print(f'Final: converged={ok}, PG={residual:.8e}, objective={value:.12e}, '
          f'iterations={len(hist)-1}, saved={output}')
    if not ok:
        raise SystemExit('The original VI tolerance was NOT attained; do not report convergence')


if __name__=='__main__':
    main()
