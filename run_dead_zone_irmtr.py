#!/usr/bin/env python3
"""Example: P1 dead-zone control, using the repository's actual IRMTR core.

Run from the root of qiwang7777/Inexact_multilevel_TR (after applying patch):

  python run_dead_zone_irmtr.py --self-test
  python run_dead_zone_irmtr.py --smoke --solver both --plot-dir dead_zone_smoke
  python run_dead_zone_irmtr.py --solver both --plots --plot-dir dead_zone_results
  python run_dead_zone_irmtr.py --replot-data dead_zone_results/dead_zone_data.json

The original repository's `IRMTRMoreau.solve_outer` contains a hard-coded
[-25,25] box and L1-specific continuation warm starts.  Those are NOT valid
for the unbounded dead-zone problem (the optimizer can exceed |z|=25).
We therefore use `IRMTRMoreau.irmtr` for *every smoothed subproblem* and
implement only the problem-specific outer Moreau continuation below.

The PDE, mesh transfers, certified inexact gradient and Galerkin coarse
Hessian logic are imported unchanged from the repository.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from optimal_control.types import Config
from optimal_control.irmtr import IRMTRMoreau
from optimal_control.pde.elliptic import (
    mesh_level, PDEMultigrid, state, smooth_value, norm, inner,
)
from optimal_control.pde.dead_zone_backend import DeadZoneEllipticBackend
from optimal_control.dead_zone_regularization import (
    dead_zone_prox, dead_zone_phi, dead_zone_moreau_value,
    dead_zone_moreau_grad, dead_zone_moreau_hessian_diagonal,
)
from optimal_control.transfer_operators import p1_transfers
from optimal_control.dead_zone_plots import make_plots
from optimal_control.dead_zone_table import (
    write_comparison_tables, write_continuation_iteration_tables,
)


def run_continuation(solver, z0, *, mu0, tau0, eps, theta_mu, theta_tau,
                     prox_r, max_outer, output_dir, method_key):
    """Problem-specific MY outer loop; inner work is original `solver.irmtr`."""
    h = len(solver.levels) - 1
    z = np.asarray(z0, dtype=float).copy()
    mu, tau = float(mu0), float(tau0)
    stages = []
    for j in range(max_outer):
        if solver.cfg.hessian_mode == 'lbfgs':
            solver.reset_lbfgs()
        start = time.perf_counter()
        z_start = z.copy()
        # THIS IS THE REPOSITORY'S TRUE RECURSIVE IRMTR SOLVER.
        result = solver.irmtr(h, z, mu, tau, inherited_delta=np.inf)
        z = result.z
        duration = time.perf_counter() - start
        upper, psi, e = solver.original_stationarity(z, prox_r, eps_hint=max(tau, 1e-12))
        F_mu = solver.objective(h, z, mu)
        record = dict(
            j=j, mu=mu, tau=tau, F_mu=float(F_mu),
            inner_iterations=int(result.iterations),
            inner_converged=bool(result.converged),
            inner_reason=str(result.reason),
            inner_final_residual=float(result.residual_bound),
            psi=float(psi), e=float(e), stationarity_upper=float(upper),
            elapsed=float(duration),
            z_min=float(z.min()), z_max=float(z.max()),
            z_L2=float(solver.norm(h, z)),
        )
        stages.append(record)
        stage_stem = f'{method_key}_stage_{j:02d}_mu_{mu:.5g}'
        np.savez_compressed(
            output_dir / f'{stage_stem}.npz',
            z=z, z_start=z_start, mu=mu, tau=tau,
            stationarity=upper, iterations=result.iterations,
        )
        nodes = solver.levels[h].mesh.nodes
        np.savetxt(output_dir/f'{stage_stem}.csv',
                   np.column_stack((nodes,z)),delimiter=',',
                   header='x1,x2,z',comments='')
        print(f' {method_key:7s} stage={j:02d} mu={mu:.4e} tau={tau:.4e} '
              f'iterations={result.iterations} inner_ok={result.converged} '
              f'F_mu={F_mu:.10e} prox+err={upper:.4e} '
              f'z=[{z.min():.4g}, {z.max():.4g}]', flush=True)
        if not result.converged:
            return z, stages, False
        if upper <= eps:
            return z, stages, True
        mu_new = theta_mu * mu
        # Correct dead-zone MY warm start; do NOT use original L1 prox.
        prox_z = dead_zone_prox(z, mu, solver.cfg.beta, solver.backend.c)
        z = prox_z + (mu_new / mu) * (z - prox_z)
        mu = mu_new
        tau *= theta_tau
    return z, stages, False


def self_test():
    rng = np.random.default_rng(20261124)
    c, beta, mu = 2., .01, .35
    v = np.array([-4., -2.001, -2., -.1, 0., .5, 2., 2.001, 3.])
    p = dead_zone_prox(v, mu, beta, c)
    assert np.all(np.abs(p) <= np.abs(v) + 1e-14)
    assert np.allclose(p[np.abs(v) <= c], v[np.abs(v) <= c])
    assert np.allclose(p[np.abs(v) > c+mu*beta],
                       v[np.abs(v) > c+mu*beta] - mu*beta*np.sign(v[np.abs(v)>c+mu*beta]))
    for n in (4, 8):
        m = mesh_level(n)
        d = rng.normal(size=len(m.nodes))
        z = rng.uniform(-4, 4, size=len(m.nodes))
        # Avoid MY transition surfaces to guarantee differentiability.
        mask = ((np.abs(z)-c < -.02) | (np.abs(z)-c > mu*beta+.02))
        z[~mask] += .1*np.sign(z[~mask])
        e = 1e-5
        fd = (dead_zone_moreau_value(m, z+e*d, mu, beta,c) -
              dead_zone_moreau_value(m, z-e*d, mu, beta,c))/(2*e)
        an = inner(m, dead_zone_moreau_grad(z,mu,beta,c), d)
        assert abs(fd-an) < 5e-8*max(1,abs(fd),abs(an)), (fd, an)
    coarse, fine = mesh_level(4),mesh_level(8)
    R,P = p1_transfers(coarse,fine)
    u=rng.normal(size=len(coarse.nodes));v=rng.normal(size=len(fine.nodes))
    assert abs(inner(fine, P@u, v)-inner(coarse, u, R@v)) < 1e-13
    assert np.allclose(R@np.ones(len(fine.nodes)), 1)
    assert np.allclose(P@np.ones(len(coarse.nodes)), 1)
    assert dead_zone_phi(fine, np.full(len(fine.nodes), 1.9), beta,c)==0
    print('PASS: dead-zone prox, MY gradient FD, P1 transfer weighted adjoint, constants.')


def json_native(obj):
    if isinstance(obj, np.generic):
        return json_native(obj.item())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: json_native(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)):
        return [json_native(t) for t in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def save_csv(path, rows, keys=None):
    if not rows:
        return
    if keys is None:
        keys = sorted({k for r in rows for k in r})
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow(json_native(r))


def main():
    parser = argparse.ArgumentParser(description='TRUE IRMTR: P1 dead-zone elliptic PDE control')
    parser.add_argument('--solver', choices=['single','multi','both'], default='both')
    parser.add_argument('--base-n', type=int, default=64,
                        help='three meshes n,2n,4n (default: 64,128,256)')
    parser.add_argument('--smoke', action='store_true',
                        help='tiny 2,4,8 mesh for integration checking')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--alpha', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--c', type=float, default=2.)
    parser.add_argument('--mu0', type=float, default=.4)
    parser.add_argument('--tau0', type=float, default=9e-5)
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--theta-mu', type=float, default=.5)
    parser.add_argument('--theta-tau', type=float, default=0.5)
    parser.add_argument('--prox-r', type=float, default=1.)
    parser.add_argument('--max-inner', type=int, default=4000)
    parser.add_argument('--max-coarse-inner', type=int, default=1)
    parser.add_argument('--max-outer', type=int, default=8)
    parser.add_argument('--fine-every', type=int, default=5)
    parser.add_argument('--kappa-ml', type=float, default=.25)
    parser.add_argument('--kappa-et', type=float, default=.2)
    parser.add_argument('--kappa-g', type=float, default=.1)
    parser.add_argument('--curvature', type=float, default=.1)
    parser.add_argument('--hessian-mode', choices=['exact','lbfgs'], default='exact')
    parser.add_argument('--memory', type=int, default=5)
    parser.add_argument('--cg-steps', type=int, default=50)
    parser.add_argument('--progress-every', type=int, default=100)
    parser.add_argument('--plot-dir', type=Path, default=Path('dead_zone_irmtr_results'))
    parser.add_argument('--plots', action='store_true',
                        help='also show matplotlib figures interactively')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--replot-data', type=Path, default=None,
                        help='render PNGs from existing results JSON/NPZ without solving')
    args = parser.parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    if args.replot_data is not None:
        data = json.loads(args.replot_data.read_text())
        make_plots(data, args.plot_dir, source_dir=args.replot_data.parent, show=args.plots)
        if data.get('results'):
            print('\n========== FINAL COMPARISON: IRMTR + MOREAU CONTINUATION ==========')
            print(write_comparison_tables(data['results'], args.plot_dir))
            print('\n========== ITERATIONS BY MOREAU PARAMETER ==========')
            print(write_continuation_iteration_tables(data['results'], args.plot_dir))
        print('Replotted data to:',args.plot_dir)
        return
    if args.self_test:
        self_test()
        return
    if args.alpha <= 0 or args.beta < 0 or args.c < 0 or args.mu0 <= 0:
        parser.error('Require alpha, mu0 positive, beta and c nonnegative')
    if not (0 < args.theta_mu < 1 and 0 < args.theta_tau < 1):
        parser.error('theta-mu and theta-tau must be in (0,1)')
    if args.smoke:
        args.base_n = 2
        args.max_inner = min(args.max_inner, 40)
        args.max_outer = min(args.max_outer, 3)

    ns = tuple(args.base_n*2**i for i in range(3))
    meshes = [mesh_level(n) for n in ns]
    print('Algorithm: ORIGINAL IRMTRMoreau.irmtr (recursive trust-region).')
    print('Dead-zone: beta * integral max(|z|-c,0), c=',args.c,'; no control box.')
    print('Control DOF:',[len(m.nodes) for m in meshes],
          'State DOF:',[len(m.interior) for m in meshes])
    print('a(x)=1+x1/2; y_d=2 sin(2pi x1) sin(pi x2)')
    print('Mass-lumped dead-zone quadrature and W-adjoint P1 transfers.')
    cfg = Config(alpha=args.alpha,beta=args.beta,
                 max_inner=args.max_inner,max_coarse_inner=args.max_coarse_inner,
                 max_outer=args.max_outer,fine_every=args.fine_every,
                 kappa_ml=args.kappa_ml,kappa_et=args.kappa_et,
                 kappa_g=args.kappa_g,curvature=args.curvature,
                 lbfgs_memory=args.memory,lbfgs_cg_maxiter=args.cg_steps,
                 hessian_mode=args.hessian_mode,verbose=not args.quiet,
                 progress_every=args.progress_every)
    z0 = np.zeros(len(meshes[-1].nodes))
    output=[]
    for method, selected, key in (
        ('Single-level TR', [meshes[-1]], 'single'),
        ('Multilevel IRMTR', meshes, 'multi'),
    ):
        if args.solver not in ('both',key):
            continue
        pde = PDEMultigrid(meshes)
        backend = DeadZoneEllipticBackend(pde,c=args.c)
        solver = IRMTRMoreau(selected,cfg,backend=backend)
        start = time.perf_counter()
        print('\n==========',method,'==========',flush=True)
        z, stages, converged = run_continuation(
            solver,z0,mu0=args.mu0,tau0=args.tau0,eps=args.eps,
            theta_mu=args.theta_mu,theta_tau=args.theta_tau,prox_r=args.prox_r,
            max_outer=args.max_outer,output_dir=args.plot_dir,method_key=key,
        )
        elapsed=time.perf_counter()-start
        finest = selected[-1]
        # Reference direct state solve only for reporting, not optimization.
        y = state(finest,z)
        fval = smooth_value(finest,z,y,args.alpha)
        phi = dead_zone_phi(finest,z,args.beta,args.c)
        upper,psi,e = solver.original_stationarity(z,args.prox_r,eps_hint=args.eps)
        fine_trace = [row for row in solver.trace if row['level']==len(selected)-1]
        summary = dict(method=method, algorithm='IRMTRMoreau.irmtr',
            converged=bool(converged), elapsed=float(elapsed),
            iterations=int(sum(row['inner_iterations'] for row in stages)),
            accepted_coarse=int(solver.counters['coarse_accepted_finest']),
            coarse_attempted=int(solver.counters['attempted']),
            rejected=int(sum(not row['accepted'] for row in fine_trace)),
            original=float(fval+phi), smooth=float(fval), phi=float(phi),
            original_prox=float(psi), prox_error=float(e),
            prox_upper=float(upper), z_min=float(z.min()),z_max=float(z.max()),
            z_L2=float(norm(finest,z)), max_gradient_error=float(solver.max_grad_error),
            gradient_evaluations=int(solver.gradient_evals),
            pde_solve_counts=json_native(pde.counts),
            **json_native(solver.counters),
        )
        print('SUMMARY:',json.dumps(summary,indent=2))
        np.savez_compressed(args.plot_dir/f'dead_zone_{key}_solution.npz',
                            z=z, y=y, nodes=finest.nodes,
                            triangles=finest.triangles, desired=2*np.sin(2*np.pi*finest.nodes[:,0])*np.sin(np.pi*finest.nodes[:,1]))
        save_csv(args.plot_dir/f'dead_zone_{key}_stages.csv',stages)
        save_csv(args.plot_dir/f'dead_zone_{key}_trace.csv',solver.trace)
        output.append(dict(**summary, key=key, stages=stages,
                           trace=json_native(solver.trace)))
    data = dict(mesh_ns=list(ns),control_dofs=[len(m.nodes) for m in meshes],
                state_dofs=[len(m.interior) for m in meshes],
                discretization='P1 state and P1 control; lumped-mass dead-zone; consistent-mass smooth part',
                config=json_native(vars(cfg)),c=float(args.c),eps=float(args.eps),
                results=output)
    (args.plot_dir/'dead_zone_data.json').write_text(json.dumps(json_native(data),indent=2))
    save_csv(args.plot_dir/'dead_zone_summary.csv', output,
             ['method','converged','iterations','elapsed','original','smooth','phi',
              'original_prox','prox_error','prox_upper','accepted_coarse',
              'coarse_attempted','rejected','z_min','z_max','z_L2',
              'gradient_evaluations','max_gradient_error'])
    make_plots(data,args.plot_dir,show=args.plots)
    print('\nSaved reports, all IRMTR trace CSVs, control arrays and figures to:', args.plot_dir)
    print('\n========== FINAL COMPARISON: IRMTR + MOREAU CONTINUATION ==========')
    print(write_comparison_tables(output, args.plot_dir))
    print('\n========== ITERATIONS BY MOREAU PARAMETER ==========')
    print(write_continuation_iteration_tables(output, args.plot_dir))
    print('\nSaved per-mu iterations: dead_zone_iterations_by_mu.txt/.md/.csv')
    print('Comparison also saved as dead_zone_comparison.txt and dead_zone_comparison.md')


if __name__ == '__main__':
    main()
