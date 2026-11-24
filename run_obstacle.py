#!/usr/bin/env python3
"""P1 obstacle indicator benchmark using ORIGINAL IRMTRMoreau.irmtr.

 python3 run_obstacle_irmtr.py --self-test
 python3 run_obstacle_irmtr.py --smoke --solver both
 python3 run_obstacle_irmtr.py --solver both --base-n 64 --plots

Variational problem (natural homogeneous Neumann boundary conditions):
 minimize_{u in H1, u>=psi} 1/2 int[(1+4 sin(2pi*x1)^2 sin(2pi*x2)^2)|grad u|² + u²] - int g u
 g(x)=10 cos(pi*x1) cos(pi*x2), psi=.1 exp(-40|x-(.5,.5)|²)-.05.

Smoothed subproblems use Moreau envelope of nodal P1 obstacle indicator
in the lumped-mass W norm. This runner does not modify irmtr.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from optimal_control.irmtr import IRMTRMoreau
from optimal_control.types import Config
from optimal_control.obstacle_mesh import mesh_level, diffusion_profile, forcing_profile
from optimal_control.transfer_operators import p1_transfers
from optimal_control.obstacle_backend import ObstacleBackend
from optimal_control.obstacle_regularization import (
    obstacle_prox, obstacle_moreau_grad, obstacle_moreau_value,
    obstacle_continuation_warm_start,
)
from optimal_control.obstacle_plots import make_plots


def native(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return native(x.item())
    if isinstance(x, dict):
        return {k: native(v) for k,v in x.items()}
    if isinstance(x, (tuple, list)):
        return [native(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def save_csv(path, rows, columns=None):
    if not rows:
        return
    if columns is None:
        columns = sorted(set().union(*(row.keys() for row in rows)))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows([native(row) for row in rows])


def pretty_table(columns, rows):
    rows = [[str(x) for x in row] for row in rows]
    widths = [max(len(columns[j]), *(len(row[j]) for row in rows))
              for j in range(len(columns))]
    def line(row):
        return ' | '.join(x.ljust(widths[j]) for j,x in enumerate(row))
    return '\n'.join([line(columns), '-+-'.join('-'*w for w in widths)] +
                     [line(row) for row in rows])


def write_reports(results, out):
    keyed = {result['key']: result for result in results}
    single, multi = keyed.get('single'), keyed.get('multi')
    columns = ['Metric', 'Single-level TR', 'Multilevel IRMTR']
    labels = [
        ('Converged','converged','boolean'),
        ('Moreau stages','stages_count','int'),
        ('Total inner iterations','iterations','int'),
        ('Elapsed time (s)','elapsed','float'),
        ('Feasible original objective','original','float'),
        ('Final projected-gradient residual','prox_upper','scientific'),
        ('Final raw infeasibility W-norm','infeasibility','scientific'),
        ('Coarse corrections attempted (all levels)','coarse_attempted','int'),
        ('Coarse steps accepted (finest level)','accepted_coarse','int'),
        ('Rejected finest-level steps','rejected','int'),
        ('Gradient evaluations (all levels)','gradient_evaluations','int'),
        ('Final active fraction (mass weighted)','active_fraction','percent'),
    ]
    def fmt(result, key, form):
        if result is None:
            return '—'
        v = result.get(key)
        if v is None:
            return '—'
        if form == 'boolean':
            return 'Yes' if v else 'No'
        if form == 'int':
            return f'{int(v):,}'
        if form == 'scientific':
            return f'{v:.4e}'
        if form == 'percent':
            return f'{100*v:.2f}%'
        return f'{v:.8f}'
    rows = [[label,fmt(single,key,form),fmt(multi,key,form)] for label,key,form in labels]
    t = pretty_table(columns,rows)
    (out/'obstacle_comparison.txt').write_text(t+'\n',encoding='utf-8')
    (out/'obstacle_comparison.md').write_text(
        '| '+' | '.join(columns)+' |\n| '+' | '.join('---' for _ in columns)+' |\n'+
        ''.join('| '+' | '.join(row)+' |\n' for row in rows),encoding='utf-8')
    print('\n========== FINAL COMPARISON: OBSTACLE IRMTR + MOREAU ==========')
    print(t)
    stages = sorted(set(round(float(stage['mu']),12)
                        for result in results for stage in result['stages']), reverse=True)
    stage_rows=[]
    csv_rows=[]
    for mu in stages:
        row_by_method={}
        for method,result in keyed.items():
            q=next((q for q in result['stages'] if abs(q['mu']-mu)<1e-10),None)
            row_by_method[method]=q
        a,b=row_by_method.get('single'),row_by_method.get('multi')
        tau=next((q['tau'] for q in (a,b) if q is not None),None)
        stage_rows.append([f'{mu:.4e}',f'{tau:.4e}' if tau is not None else '—',
                           str(a['inner_iterations']) if a is not None else '—',
                           str(b['inner_iterations']) if b is not None else '—'])
        csv_rows.append(dict(mu=mu,tau=tau,
                             single_iterations=a['inner_iterations'] if a else None,
                             multi_iterations=b['inner_iterations'] if b else None,
                             single_prox_residual=a['stationarity_upper'] if a else None,
                             multi_prox_residual=b['stationarity_upper'] if b else None))
    stage_rows.append(['TOTAL','—',
                       str(single['iterations']) if single is not None else '—',
                       str(multi['iterations']) if multi is not None else '—'])
    t2=pretty_table(['mu','tau','Single-level TR','Multilevel IRMTR'],stage_rows)
    print('\n========== ITERATIONS BY MOREAU PARAMETER ==========')
    print(t2)
    (out/'obstacle_iterations_by_mu.txt').write_text(t2+'\n',encoding='utf-8')
    save_csv(out/'obstacle_iterations_by_mu.csv',csv_rows)
    save_csv(out/'obstacle_summary.csv',[
        {k:v for k,v in q.items() if k not in ('stages','trace')}
        for q in results])


def run_continuation(solver, z0, *, mu0, tau0, eps, theta_mu,
                     theta_tau, prox_r, max_outer, out, key):
    h=len(solver.levels)-1
    mesh=solver.levels[-1].mesh
    psi=solver.backend.obstacle(mesh)
    z=np.asarray(z0,dtype=float).copy()
    mu,tau=float(mu0),float(tau0)
    stages=[]
    for j in range(max_outer):
        if solver.cfg.hessian_mode=='lbfgs':
            solver.reset_lbfgs()
        z_start=z.copy()
        start=time.perf_counter()
        result=solver.irmtr(h,z,mu,tau,inherited_delta=np.inf)
        z=np.asarray(result.z,dtype=float)
        duration=time.perf_counter()-start
        feasible=obstacle_prox(z,psi)
        violation=solver.norm(h,z-feasible)
        upper,prox_resid,grad_error=solver.original_stationarity(
            feasible,prox_r,eps_hint=max(tau,1e-12))
        active=np.abs(feasible-psi)<=1e-6
        active_fraction=float(np.dot(mesh.W,active.astype(float))/mesh.W.sum())
        Fmu=float(solver.objective(h,z,mu))
        J=float(solver.backend.smooth_value(mesh,feasible))
        rec=dict(j=j,mu=mu,tau=tau,inner_iterations=int(result.iterations),
                 inner_converged=bool(result.converged),inner_reason=str(result.reason),
                 inner_final_residual=float(result.residual_bound),
                 stationarity_upper=float(upper),prox_residual=float(prox_resid),
                 gradient_error=float(grad_error),infeasibility=float(violation),
                 F_mu=Fmu,original_projected=J,active_fraction=active_fraction,
                 elapsed=duration,z_min=float(feasible.min()),z_max=float(feasible.max()),
                 z_L2=float(solver.norm(h,feasible)))
        stages.append(rec)
        np.savez_compressed(out/f'{key}_stage_{j:02d}_mu_{mu:.5g}.npz',
                            raw=z,feasible=feasible,initial=z_start,mu=mu,tau=tau)
        print(f' {key:7s} stage={j:02d} mu={mu:.3e} tau={tau:.3e} '
              f'iterations={result.iterations} inner_ok={result.converged} '
              f'F_mu={Fmu:.8e} J_feas={J:.8e} '
              f'PG={upper:.3e} violation={violation:.3e}',flush=True)
        if not result.converged:
            return feasible,z,stages,False
        if upper<=eps:
            return feasible,z,stages,True
        mu_new=theta_mu*mu
        z=obstacle_continuation_warm_start(z,psi,mu,mu_new)
        mu=mu_new
        tau*=theta_tau
    return obstacle_prox(z,psi),z,stages,False


def self_test():
    from optimal_control.obstacle_regularization import (
        obstacle_moreau_hessian_diagonal, obstacle_profile, obstacle_phi)
    rng=np.random.default_rng(2701)
    mesh=mesh_level(4)
    bkd=ObstacleBackend([mesh])
    psi=bkd.obstacle(mesh)
    assert obstacle_phi(obstacle_prox(psi-1,psi),psi)==0
    assert np.isinf(obstacle_phi(psi-.1,psi))
    mu=.3
    z=psi+rng.choice([-.12,.15],size=len(psi))
    v=rng.normal(size=len(psi));v/=np.linalg.norm(v)
    h=1e-6
    val=lambda x: obstacle_moreau_value(mesh.W,x,psi,mu)
    deriv=(val(z+h*v)-val(z-h*v))/(2*h)
    analytic=float(np.dot(mesh.W*obstacle_moreau_grad(z,psi,mu),v))
    assert abs(deriv-analytic)<1e-9, (deriv,analytic)
    diagonal=obstacle_moreau_hessian_diagonal(z,psi,mu)
    g=lambda x: obstacle_moreau_grad(x,psi,mu)
    Hv_fd=(g(z+h*v)-g(z-h*v))/(2*h)
    assert np.allclose(diagonal*v,Hv_fd,rtol=1e-8,atol=1e-9)
    solver=IRMTRMoreau([mesh],Config(alpha=0,beta=0),backend=bkd)
    H=solver.make_model(0,z,mu)
    grad=lambda x: bkd.smooth_gradient(mesh,x)+obstacle_moreau_grad(x,psi,mu)
    Hv_fd=(grad(z+h*v)-grad(z-h*v))/(2*h)
    Hv=H.matvec(v)
    assert np.linalg.norm(Hv-Hv_fd)/max(1e-12,np.linalg.norm(Hv))<1e-8
    w=rng.normal(size=len(v))
    sym=abs(bkd.inner(mesh,v,H.matvec(w))-bkd.inner(mesh,Hv,w))
    assert sym<1e-10
    fine=mesh_level(8)
    R,P=p1_transfers(mesh,fine)
    x=rng.normal(size=len(mesh.nodes))
    y=rng.normal(size=len(fine.nodes))
    err=abs(bkd.inner(fine,P@x,y)-bkd.inner(mesh,x,R@y))
    assert err<1e-12
    bkd2=ObstacleBackend([mesh,fine])
    solver2=IRMTRMoreau([mesh,fine],Config(alpha=0,beta=0),backend=bkd2)
    zf=bkd2.obstacle(fine)-rng.choice([.07,-.13],size=len(fine.nodes))
    Hf=solver2.make_model(1,zf,mu)
    d=rng.normal(size=len(mesh.nodes))
    Bc_d=R@Hf.matvec(P@d)
    energy_c=bkd.inner(mesh,d,Bc_d)
    energy_f=bkd.inner(fine,P@d,Hf.matvec(P@d))
    assert abs(energy_c-energy_f)<1e-9
    # MY gradient-preserving continuation warm start
    zf0=zf.copy()
    zf1=obstacle_continuation_warm_start(zf0,bkd2.obstacle(fine),mu,mu/2)
    assert np.allclose(obstacle_moreau_grad(zf0,bkd2.obstacle(fine),mu),
                       obstacle_moreau_grad(zf1,bkd2.obstacle(fine),mu/2))
    print('PASS: feasible projection, Moreau FD gradient/Hessian, full Hessian-vector FD,')
    print('      W symmetry, P1 weighted adjoint, Galerkin curvature, continuation warm start.')


def main():
    ap=argparse.ArgumentParser(description='P1 obstacle indicator using original IRMTRMoreau.irmtr')
    ap.add_argument('--solver',choices=['single','multi','both'],default='both')
    ap.add_argument('--base-n',type=int,default=64,help='3 meshes n,2n,4n; default 64 gives 4225,16641,66049 DOF')
    ap.add_argument('--mu0',type=float,default=.2)
    ap.add_argument('--tau0',type=float,default=1e-3)
    ap.add_argument('--eps',type=float,default=1e-4)
    ap.add_argument('--theta-mu',type=float,default=.5)
    ap.add_argument('--theta-tau',type=float,default=.5)
    ap.add_argument('--prox-r',type=float,default=.05,
                    help='positive projected-gradient step; 0.05 is suited to this elliptic energy')
    ap.add_argument('--max-inner',type=int,default=1500)
    ap.add_argument('--max-coarse-inner',type=int,default=1)
    ap.add_argument('--max-outer',type=int,default=12)
    ap.add_argument('--fine-every',type=int,default=5)
    ap.add_argument('--kappa-ml',type=float,default=.25)
    ap.add_argument('--kappa-et',type=float,default=.2)
    ap.add_argument('--kappa-g',type=float,default=.1)
    ap.add_argument('--hessian-mode',choices=['exact','lbfgs'],default='exact')
    ap.add_argument('--memory',type=int,default=5)
    ap.add_argument('--cg-steps',type=int,default=50)
    ap.add_argument('--progress-every',type=int,default=100)
    ap.add_argument('--plot-dir',type=Path,default=Path('obstacle_irmtr_results'))
    ap.add_argument('--plots',action='store_true',help='show plots interactively')
    ap.add_argument('--quiet',action='store_true')
    ap.add_argument('--smoke',action='store_true',help='mesh hierarchy 2,4,8 for integration test')
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--replot-data',type=Path,default=None)
    args=ap.parse_args()
    if args.self_test:
        self_test()
        return
    out=args.plot_dir
    out.mkdir(parents=True,exist_ok=True)
    if args.replot_data is not None:
        data=json.loads(args.replot_data.read_text(encoding='utf-8'))
        make_plots(data,out,source_dir=args.replot_data.parent,show=args.plots)
        write_reports(data['results'],out)
        return
    if args.mu0<=0 or args.tau0<=0 or args.eps<=0 or args.prox_r<=0:
        ap.error('mu0, tau0, eps, prox-r must be positive')
    if not(0<args.theta_mu<1 and 0<args.theta_tau<1):
        ap.error('theta-mu and theta-tau must lie between 0 and 1')
    if args.smoke:
        args.base_n=2
        args.max_inner=min(100,args.max_inner)
        args.max_outer=min(3,args.max_outer)
    ns=tuple(args.base_n*2**j for j in range(3))
    meshes=[mesh_level(n) for n in ns]
    print('Algorithm: ORIGINAL IRMTRMoreau.irmtr, exact Galerkin Bc=R Bf P')
    print('Obstacle problem: reaction-diffusion energy, natural Neumann BC, nodal P1 u>=psi')
    print('a(x) = 1+4 sin(2pi*x1)^2 sin(2pi*x2)^2; g(x) = 10 cos(pi*x1) cos(pi*x2)')
    print('Mesh square divisions:',list(ns),'Total nodal DOF:',[len(m.nodes) for m in meshes])
    cfg=Config(alpha=0.,beta=0.,max_inner=args.max_inner,
               max_coarse_inner=args.max_coarse_inner,max_outer=args.max_outer,
               fine_every=args.fine_every,kappa_ml=args.kappa_ml,
               kappa_et=args.kappa_et,kappa_g=args.kappa_g,
               hessian_mode=args.hessian_mode,lbfgs_memory=args.memory,
               lbfgs_cg_maxiter=args.cg_steps,verbose=not args.quiet,
               progress_every=args.progress_every)
    z0=np.zeros(len(meshes[-1].nodes))
    results=[]
    for name,selected,key in (
        ('Single-level TR',[meshes[-1]],'single'),
        ('Multilevel IRMTR',meshes,'multi')):
        if args.solver not in ('both',key):
            continue
        backend=ObstacleBackend(selected)
        solver=IRMTRMoreau(selected,cfg,backend=backend)
        start=time.perf_counter()
        print('\n==========',name,'==========',flush=True)
        feasible,raw,stages,converged=run_continuation(
            solver,z0,mu0=args.mu0,tau0=args.tau0,eps=args.eps,
            theta_mu=args.theta_mu,theta_tau=args.theta_tau,
            prox_r=args.prox_r,max_outer=args.max_outer,out=out,key=key)
        elapsed=time.perf_counter()-start
        mesh=selected[-1]
        psi=backend.obstacle(mesh)
        fine_trace=[row for row in solver.trace if row['level']==len(selected)-1]
        upper,_,_=solver.original_stationarity(feasible,args.prox_r,eps_hint=args.eps)
        mass_active=float(np.dot(mesh.W,(np.abs(feasible-psi)<1e-6))/mesh.W.sum())
        summary=dict(key=key,method=name,algorithm='IRMTRMoreau.irmtr',
                     converged=converged,stages_count=len(stages),
                     iterations=sum(q['inner_iterations'] for q in stages),
                     elapsed=float(elapsed),original=backend.smooth_value(mesh,feasible),
                     prox_upper=float(upper),
                     infeasibility=float(solver.norm(len(selected)-1,raw-feasible)),
                     active_fraction=mass_active,
                     accepted_coarse=int(solver.counters['coarse_accepted_finest']),
                     coarse_attempted=int(solver.counters['attempted']),
                     rejected=sum(not row['accepted'] for row in fine_trace),
                     gradient_evaluations=solver.gradient_evals,
                     counters=native(solver.counters),stages=stages,trace=native(solver.trace))
        print('SUMMARY:',json.dumps(native({k:v for k,v in summary.items() if k!='trace'}),indent=2))
        np.savez_compressed(out/f'obstacle_{key}_solution.npz',
                            z=feasible,z_raw=raw,psi=psi,
                            forcing=forcing_profile(mesh.nodes), diffusion=mesh.a_nodes,
                            nodes=mesh.nodes,triangles=mesh.triangles)
        save_csv(out/f'obstacle_{key}_stages.csv',stages)
        save_csv(out/f'obstacle_{key}_trace.csv',solver.trace)
        results.append(summary)
    data=dict(mesh_ns=list(ns),node_dofs=[len(m.nodes) for m in meshes],
              epsilon=args.eps,eps=args.eps,config=native(vars(cfg)),
              formulation='Neumann obstacle variational inequality, P1 nodes, W-metric Moreau indicator',
              results=results)
    (out/'obstacle_data.json').write_text(json.dumps(native(data),indent=2),encoding='utf-8')
    write_reports(results,out)
    make_plots(data,out,show=args.plots)
    print('\nSaved reports, per-mu histories, fields, and plots to:',out)

if __name__=='__main__':
    main()
