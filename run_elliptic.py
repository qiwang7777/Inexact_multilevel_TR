#!/usr/bin/env python3
"""Example 2: linear elliptic equation, P1 controls, certified inexact gradients.

Usage from repository root:
  python3 run_elliptic.py --self-test
  python3 run_elliptic.py --smoke --plot-dir example2_smoke
  python3 run_elliptic.py --solver both --plot-dir example2_results
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np

from optimal_control.pde.elliptic import (
    FEMLevel, mesh_level, state, adjoint, smooth_value, smooth_gradient,
    inner, norm, PDEMultigrid)
from optimal_control.transfer_operators import p1_transfers as transfers
from optimal_control.irmtr import IRMTRMoreau
from optimal_control.types import Config
from optimal_control.pde.elliptic_backend import EllipticBackend
from optimal_control.regularization import (
    prox_phi, moreau_value, moreau_grad, moreau_hessian_diagonal)
from optimal_control.reporting import (
    elliptic_original_values as original_values,
    report_elliptic_results as report_results)
from optimal_control.plots import plot_elliptic_results as plots

def self_test():
    c=mesh_level(4);f=mesh_level(8)
    R,P=transfers(c,f)
    rng=np.random.default_rng(1234)
    u=rng.standard_normal(len(c.nodes));v=rng.standard_normal(len(f.nodes))
    adj_defect=abs(inner(f,P@u,v)-inner(c,u,R@v))
    assert adj_defect<1e-13,adj_defect
    assert np.allclose(R@np.ones(len(f.nodes)),1.)
    assert np.allclose(P@np.ones(len(c.nodes)),1.)
    assert norm(f,P@u)<=norm(c,u)*(1+1e-13)
    assert np.linalg.norm(R@P@u-u)>1e-6,'RP is not identity for P1'
    ic=jc=2
    cr=ic*(c.n+1)+jc
    stencil=np.array([[R[cr,(2*ic+di)*(f.n+1)+(2*jc+dj)]
                      for di in [-1,0,1]] for dj in [1,0,-1]])
    target=np.array([[0,1,1],[1,2,1],[1,1,0]],dtype=float)/8
    assert np.allclose(stencil,target),f'wrong seven-point stencil: {stencil}'
    assert R.getrow(cr).nnz==7
    z=.2+.2*rng.standard_normal(len(f.nodes))
    d=rng.standard_normal(len(f.nodes))
    h=1e-5
    y=state(f,z);p=adjoint(f,y)
    g=smooth_gradient(f,z,p,1e-4)
    fd=(smooth_value(f,z+h*d,state(f,z+h*d),1e-4)-
        smooth_value(f,z-h*d,state(f,z-h*d),1e-4))/(2*h)
    deriv=inner(f,g,d)
    assert abs(fd-deriv)<1e-7*max(1,abs(fd),abs(deriv)),(fd,deriv)
    mu=.05
    fdm=(moreau_value(f,z+h*d,mu,.01)-
         moreau_value(f,z-h*d,mu,.01))/(2*h)
    gm=inner(f,moreau_grad(z,mu,.01),d)
    assert abs(fdm-gm)<1e-7*max(1,abs(fdm),abs(gm)),(fdm,gm)
    # PDE must include the variable coefficient and the reaction term.
    assert abs(f.A[0,0]-f.M[0,0])>1
    # Rigorous residual-based gradient estimate must bound the actual
    # difference to the fully solved discrete state/adjoint reference.
    ms=[mesh_level(4),mesh_level(8)]
    solver=PDEMultigrid(ms)
    zz=rng.uniform(-2,2,len(ms[-1].nodes))
    yy,ry=solver.state(1,zz)
    pp,ee,_=solver.adjoint(1,yy,5e-4,ry)
    gg=smooth_gradient(ms[-1],zz,pp,1e-4)
    ystar=state(ms[-1],zz)
    pstar=adjoint(ms[-1],ystar)
    gstar=smooth_gradient(ms[-1],zz,pstar,1e-4)
    error=norm(ms[-1],gg-gstar)
    assert error<=ee*(1+1e-8),(error,ee)
    assert ee>0
    print('Self-test PASSED: weighted adjoint, P/R constants, seven-point '
          'stencil, P1 gradients, MY gradients, elliptic assembly.')
    print('Interior restriction stencil:\n',stencil)
    print(f'Smooth gradient FD={fd:.10e}, derivative={deriv:.10e}')

def main():
    p=argparse.ArgumentParser(description='Example 2: P1 state/control variable-diffusion PDE, single TR vs IRMTR')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--replot-data',type=Path,default=None,
                   help='Recreate figures from a saved example2_comparison_data.json without solving PDEs')
    p.add_argument('--smoke',action='store_true',help='Use n=(2,4,8), fewer iterations')
    p.add_argument('--base-n',type=int,default=32,help='Three meshes n=(n,2n,4n)')
    p.add_argument('--solver',choices=['both','single','multi'],default='both')
    p.add_argument('--mu',type=float,default=None,help='If set: solve only the fixed-mu smoothed problem')
    p.add_argument('--tau',type=float,default=.002,help='Fixed-mu inner tolerance')
    p.add_argument('--alpha',type=float,default=1e-4)
    p.add_argument('--beta',type=float,default=0.0)
    p.add_argument('--mu0',type=float,default=.1)
    p.add_argument('--tau0',type=float,default=.008)
    p.add_argument('--eps',type=float,default=.001)
    p.add_argument('--max-inner',type=int,default=4000)
    p.add_argument('--max-coarse-inner',type=int,default=1)
    p.add_argument('--max-outer',type=int,default=8)
    p.add_argument('--fine-every',type=int,default=5)
    p.add_argument('--kappa-ml',type=float,default=.25)
    p.add_argument('--kappa-et',type=float,default=.2)
    p.add_argument('--curvature',type=float,default=.1)
    p.add_argument('--kappa-g',type=float,default=.10,help='Certified dynamic gradient forcing parameter')
    p.add_argument('--memory',type=int,default=5)
    p.add_argument('--hessian-mode',choices=['exact','lbfgs'],default='exact')
    p.add_argument('--cg-steps',type=int,default=50)
    p.add_argument('--plot-dir',type=Path,default=Path('example2_results'))
    p.add_argument('--quiet',action='store_true')
    p.add_argument('--progress-every',type=int,default=100)
    p.add_argument('--compare-mus',nargs='*',type=float,default=None)
    args=p.parse_args()
    if not (0 < args.kappa_g < 1):
        p.error('--kappa-g must be in (0,1)')
    if args.self_test:
        self_test();return
    if args.replot_data is not None:
        data=json.loads(args.replot_data.read_text())
        args.plot_dir.mkdir(parents=True,exist_ok=True)
        plots(data['results'],args.plot_dir,args.eps,args.compare_mus)
        report_results(data['results'],args.plot_dir)
        print('Recreated plots and reporting tables from',args.replot_data)
        return
    if args.smoke:
        args.base_n=2;args.max_inner=min(args.max_inner,60)
        args.max_outer=min(args.max_outer,3)
    ns=(args.base_n,2*args.base_n,4*args.base_n)
    meshes=[mesh_level(n) for n in ns]
    outdir=args.plot_dir
    outdir.mkdir(parents=True,exist_ok=True)
    if args.alpha <= 0 or args.beta < 0:
        p.error('Require alpha > 0 and beta >= 0')
    cfg=Config(alpha=args.alpha, beta=args.beta,
               max_inner=args.max_inner,max_coarse_inner=args.max_coarse_inner,
               max_outer=args.max_outer,fine_every=args.fine_every,
               kappa_ml=args.kappa_ml,kappa_et=args.kappa_et,
               curvature=args.curvature,
               lbfgs_memory=args.memory,
               lbfgs_cg_maxiter=args.cg_steps,
               verbose=not args.quiet,progress_every=args.progress_every,
               kappa_g=args.kappa_g,hessian_mode=args.hessian_mode)
    h=len(meshes)-1
    z0=np.zeros(len(meshes[-1].nodes))
    print('PDE: -div((1 + x1/2) grad y) + y = z, y|boundary=0')
    print('Desired state: y_d=2 sin(2 pi x1) sin(pi x2)')
    print('Hessian model:', cfg.hessian_mode)
    print('Mesh intervals:',ns)
    print('P1 control DOFs:',[len(m.nodes) for m in meshes])
    print('P1 state DOFs:',[len(m.interior) for m in meshes])
    print('Transfer: nodal P, R=Wc^-1 P^T Wf; mass-lumped inner products')
    print(f'alpha={cfg.alpha:g}, beta={cfg.beta:g}, box=[-25,25]')
    print(f'Gradient solves: PCG + geometric multigrid; certified dynamic error, kappa_g={cfg.kappa_g:g}')
    rows=[]
    for method,selected in [('Single-level TR',[meshes[-1]]),
                             ('Multilevel IRMTR',meshes)]:
        if args.solver=='single' and method!='Single-level TR':continue
        if args.solver=='multi' and method!='Multilevel IRMTR':continue
        pde=PDEMultigrid(meshes)
        s=IRMTRMoreau(selected, cfg, backend=EllipticBackend(pde))
        start=time.perf_counter()
        print('\n==========',method,'==========',flush=True)
        if args.mu is None:
            result=s.solve_outer(z0,args.mu0,args.tau0,args.eps)
            stages=result.stages
            z=result.z;converged=result.converged
            last_mu=stages[-1]['mu'] if stages else args.mu0
        else:
            s.reset_lbfgs()
            inn=s.irmtr(len(selected)-1,z0,args.mu,args.tau)
            z=inn.z;converged=inn.converged;last_mu=args.mu
            stages=[dict(j=0,mu=args.mu,tau=args.tau,
                         inner_iterations=inn.iterations,inner_converged=inn.converged,
                         inner_final_residual=inn.residual,
                         inner_reason=inn.reason,
                         stationarity_upper=s.original_stationarity(z,1.,eps_hint=args.eps)[0],
                         elapsed=time.perf_counter()-start)]
        elapsed=time.perf_counter()-start
        values=original_values(selected[-1],z,last_mu,cfg.alpha,cfg.beta,pde)
        _,psi,psi_e=s.original_stationarity(z,1.,eps_hint=args.eps)
        z_feas=np.clip(z,-25.0,25.0)
        _,feasible_psi,feasible_e=s.original_stationarity(z_feas,1.,eps_hint=args.eps)
        fine_trace=[q for q in s.trace if q['level']==len(selected)-1]
        summary=dict(method=method,converged=converged,elapsed=elapsed,
                     finest=len(selected)-1,
                     iterations=sum(q['inner_iterations'] for q in stages),
                     rejected=sum(not q['accepted'] for q in fine_trace),
                     finest_coarse_candidates=sum(q['mode']=='coarse' for q in fine_trace),
                     finest_coarse_accepted=s.counters['coarse_accepted_finest'],
                     psi=float(psi),psi_error=float(psi_e),psi_plus_e=float(psi+psi_e),
                     feasible_psi=float(feasible_psi),
                     feasible_error=float(feasible_e),
                     feasible_upper=float(feasible_psi+feasible_e),
                     max_gradient_error=s.max_grad_error,
                     max_dynamic_error_ratio=s.max_grad_ratio,
                     gradient_evaluations=s.gradient_evals,
                     pde_solve_counts=pde.counts.copy(),
                     **values,counters=s.counters)
        print('Result:',json.dumps(summary,indent=2,allow_nan=False))
        # Outer records retain the actual P1 control arrays for plotting.
        # The publication JSON only needs scalar continuation diagnostics.
        stage_records=[{k:v for k,v in row.items()
                        if k not in ('z_start','z_end')}
                       for row in stages]
        trace_records=[]
        for rec in s.trace:
            clean={}
            for k,v in rec.items():
                if isinstance(v,(float,np.floating)) and not np.isfinite(v):
                    clean[k]=None
                elif isinstance(v,np.generic):
                    clean[k]=v.item()
                else:
                    clean[k]=v
            trace_records.append(clean)
        rows.append(dict(**summary,stages=stage_records,trace=trace_records))
        np.savez_compressed(outdir/('example2_'+('single' if len(selected)==1 else 'multi')+'_solution.npz'),
                            z=z, nodes=selected[-1].nodes, triangles=selected[-1].triangles)
    data=dict(mesh_ns=ns,control_dofs=[len(m.nodes) for m in meshes],
              state_dofs=[len(m.interior) for m in meshes],
              config=vars(cfg),results=rows)
    (outdir/'example2_comparison_data.json').write_text(json.dumps(data,indent=2,allow_nan=False))
    with (outdir/'example2_summary.csv').open('w',newline='') as f:
        keys=['method','converged','iterations','elapsed','rejected',
              'finest_coarse_candidates','finest_coarse_accepted','smoothed',
              'original','projected','max_box_violation','psi','psi_error','psi_plus_e',
              'feasible_psi','feasible_error','feasible_upper',
              'max_gradient_error','gradient_evaluations']
        wr=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
        wr.writeheader();wr.writerows(rows)
    report_results(rows,outdir)
    if args.mu is None and len(rows)==2:
        plots(rows,outdir,args.eps,args.compare_mus)
    print('\nSaved results under',outdir.resolve())
    if args.mu is not None:
        print('Fixed-mu converged means stationarity of SMOOTHED F_mu, not original F.')
    if any(not row['converged'] for row in rows):
        print('WARNING: At least one solver did not meet its stated convergence test.')


if __name__=='__main__':
    main()
