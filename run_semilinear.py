"""Run the original semilinear example using the modular optimal_control package.

Execute with `python run_semilinear.py --solver both`.
"""

import argparse
import time
import numpy as np

from optimal_control.geometry import create_unit_square_mesh
from optimal_control.types import Config
from optimal_control.irmtr import IRMTRMoreau
from optimal_control.validation import check_exact_hessian, check_moreau_curvature
from optimal_control.reporting import export_stage_controls, report_solution
from optimal_control.plots import (plot_results, plot_z_by_mu_compare,
                                   plot_continuation_history_compare,
                                   plot_final_controls_vs_reference)


def main():
    parser = argparse.ArgumentParser(
        description='FULL nonsmooth semilinear PDE control: exact Hessian + MY continuation'
    )
    parser.add_argument('--base-n', type=int, default=8,
                        help='4 grids: n,2n,4n,8n (default 8,16,32,64)')
    parser.add_argument('--full', action='store_true',
                        help='Very large meshes (64,128,256,512); costly')
    parser.add_argument('--solver', choices=['both','single','multi'], default='both')
    parser.add_argument('--hessian-mode', choices=['exact','lbfgs','diagonal'], default='exact')
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--alpha', type=float, default=1e-4)
    parser.add_argument('--box', type=float, default=25.0,
                        help='Current implementation uses [-25,25]; other values unsupported')
    parser.add_argument('--mu0', type=float, default=0.4)
    parser.add_argument('--tau-fixed', '--tau0', dest='tau_fixed',
                        type=float, default=8e-5,
                        help='Initial inner tolerance; decreases with theta-mu; must be < eps. '
                             '--tau0 accepted as a backward-compatible alias')
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--theta-mu', type=float, default=0.5,
                        help='Decrease mu and tau each outer stage')
    parser.add_argument('--prox-r', type=float, default=1.0)
    parser.add_argument('--max-outer', type=int, default=8)
    parser.add_argument('--max-inner', type=int, default=4000)
    parser.add_argument('--max-coarse-inner', type=int, default=1)
    parser.add_argument('--kappa-g', type=float, default=0.1,
                        help='Certified dynamic gradient forcing parameter')
    parser.add_argument('--cg-maxiter', type=int, default=50)
    parser.add_argument('--cg-rtol', type=float, default=1e-3)
    parser.add_argument('--kappa-ml', type=float, default=0.15)
    parser.add_argument('--kappa-et', type=float, default=0.2)
    parser.add_argument('--progress-every', type=int, default=15)
    parser.add_argument('--lbfgs-memory', type=int, default=5)
    parser.add_argument('--fixed-mu', action='store_true',
                        help='Optional diagnostic: one fixed-mu full MY subproblem, no outer loop')
    parser.add_argument('--smooth-only', action='store_true',
                        help='Optional diagnostic ONLY: omit MY; default includes it')
    parser.add_argument('--hessian-check', action='store_true',
                        help='Small-mesh Hessian-vector finite-difference/symmetry check')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--plots', action='store_true', help='Display Spyder/Matplotlib plots')
    parser.add_argument('--print-full-z', action='store_true',
                        help='Print ALL P0 z coefficients after every outer mu stage (very long)')
    parser.add_argument('--no-export-z', action='store_true',
                        help='Disable stage-by-stage CSV/NPZ control export')
    parser.add_argument('--z-output-dir', type=str, default='optimal_z_by_mu',
                        help='Directory for output control vectors (relative to working directory)')
    args, _unknown = parser.parse_known_args()  # compatible with Spyder runfile
    if args.box != 25.0:
        parser.error('This script currently implements the fixed box [-25,25].')
    if args.smooth_only and not args.fixed_mu:
        parser.error('--smooth-only is a diagnostic for --fixed-mu; full mode includes MY.')
    if args.beta < 0 or args.alpha <= 0:
        parser.error('beta must be nonnegative and alpha positive')
    if not (0 < args.kappa_g < 1):
        parser.error('--kappa-g must be in (0,1)')
    if args.cg_maxiter <= 0 or args.max_inner <= 0 or args.max_outer <= 0:
        parser.error('Iteration limits must be positive')
    if not (args.eps > args.tau_fixed > 0):
        parser.error('Require 0 < --tau-fixed < --eps (e.g. 1e-4 < 1e-3)')
    ns = (64,128,256,512) if args.full else tuple(args.base_n * 2**i for i in range(4))
    meshes = [create_unit_square_mesh(n) for n in ns]
    finest = meshes[-1]
    print('Mesh n:', ns)
    print('Control DOFs:', [len(m.elements) for m in meshes])
    print('\n========== FULL MOREAU-YOSIDA MODEL ==========' if not args.smooth_only
          else '\n========== SMOOTH-ONLY DIAGNOSTIC ==========')
    print('Objective: f(z) + M_mu[beta*||z||_1 + I_{[-25,25]}](z)')
    print('PDE reduced Hessian:', args.hessian_mode)
    print('Gradient error: discrete PDE-residual certificate, kappa_g:', args.kappa_g)
    print('Moreau regularization: enabled' if not args.smooth_only else
          'Moreau regularization: DISABLED (diagnostic)')
    print(f'beta={args.beta}, alpha={args.alpha}')
    print(f'outer mu0={args.mu0}, tau0={args.tau_fixed}, eps={args.eps}, max_outer={args.max_outer}')

    def make_config():
        return Config(alpha=args.alpha, beta=args.beta,
                      desired_state=-1.0,
                      smooth_only=args.smooth_only,
                      hessian_mode=args.hessian_mode,
                      lbfgs_memory=args.lbfgs_memory,
                      lbfgs_cg_maxiter=args.cg_maxiter,
                      lbfgs_cg_rtol=args.cg_rtol,
                      kappa_g=args.kappa_g,
                      kappa_ml=args.kappa_ml,
                      kappa_et=args.kappa_et,
                      max_inner=args.max_inner,
                      max_coarse_inner=args.max_coarse_inner,
                      max_outer=args.max_outer,
                      progress_every=args.progress_every,
                      verbose=not args.quiet)

    if args.hessian_check and args.hessian_mode == 'exact':
        print('\n========== EXACT + MY HESSIAN CHECK (SMOOTH REGIONS) ==========')
        test_mesh = meshes[0]
        test_z = 2.0 * np.ones(len(test_mesh.elements))
        # With beta>0, soft region is |z| < mu*beta, box region is |z|>25+mu*beta;
        # at z=2 we are away from either switching surface.
        check_exact_hessian(test_mesh, make_config(), test_z, args.mu0)
        if not args.smooth_only:
            check_moreau_curvature(args.mu0, args.beta)

    z0 = 2.0*np.ones(len(finest.elements))
    solvers = []
    if args.solver in ('both','single'):
        solvers.append(('Single-level TR', [finest]))
    if args.solver in ('both','multi'):
        solvers.append(('Multilevel IRMTR', meshes))
    results=[]
    outer_outputs={}
    for name, selected_meshes in solvers:
        solver = IRMTRMoreau(selected_meshes, make_config())
        h = len(solver.levels)-1
        print(f'\n========== {name.upper()} ==========')
        start = time.perf_counter()
        if args.fixed_mu:
            inner = solver.irmtr(h, z0.copy(), args.mu0, args.tau_fixed)
            elapsed = time.perf_counter()-start
            ev = solver.evaluate(h, inner.z, args.mu0, args.tau_fixed)
            total, psi, e = solver.original_stationarity(inner.z, args.prox_r, args.tau_fixed)
            rows=[row for row in solver.history if row['level']==h]
            print(f'Fixed mu={args.mu0}, tau={args.tau_fixed} (NO outer continuation)')
            print('Inner converged:', inner.converged, 'reason:', inner.reason)
            print('Inner iterations:', inner.iterations)
            print('Smoothed objective:', f'{ev.F:.12e}')
            print('Smoothed gradient norm:', f'{solver.norm(h, ev.G):.9e}')
            print('Original proximal residual:', f'{psi:.9e}')
            print('Accepted coarse corrections:',
                  sum(row['mode']=='coarse' and row['accepted'] for row in rows))
            print('Rejected steps:', sum(not row['accepted'] for row in rows))
            print('Elapsed seconds:', f'{elapsed:.3f}')
            results.append(dict(name=name, time=elapsed, iterations=inner.iterations,
                                converged=inner.converged, prox=psi))
        else:
            out=solver.solve_outer(z0.copy(), mu0=args.mu0, tau_fixed=args.tau_fixed,
                                   eps=args.eps, theta_mu=args.theta_mu,
                                   r=args.prox_r, max_outer=args.max_outer,
                                   report_z=True, print_full_z=args.print_full_z)
            elapsed=time.perf_counter()-start
            outer_outputs[name] = out
            results.append(report_solution(name, solver, out, elapsed,
                                           args.prox_r, args.eps))
            if not args.no_export_z:
                print(f'\n========== {name}: CONTROL VECTORS BY mu ==========')
                export_stage_controls(finest, out, name, args.z_output_dir)
            if args.plots:
                plot_results(finest, solver, out, eps=args.eps)
            print('PDE and exact-Hessian work by level:')
            for level, work in solver.cost.items():
                print(f"  level {level}: evaluations={work['eval_calls']}, "
                      f"trial objectives={work['obj_calls']}, "
                      f"reused trial states={work['reused_trial_states']}, "
                      f"Hessian builds={work['exact_hessian_builds']}, "
                      f"Hessian-vector products={work['exact_hessian_products']}")
    print('\n========== COMPARISON ==========')
    for r in results:
        print(f"{r['name']}: iterations={r['iterations']}, "
              f"time={r['time']:.3f}s, converged={r['converged']}, "
              f"original_prox={r['prox']:.3e}")
        
    if ('Single-level TR' in outer_outputs and 'Multilevel IRMTR' in outer_outputs):
        out_single = outer_outputs['Single-level TR']
        out_multi = outer_outputs['Multilevel IRMTR']
        plot_z_by_mu_compare(finest,out_single,out_multi,lb=-25.0,ub=25.0,clip_to_box=True)
        plot_continuation_history_compare(out_single,out_multi, eps=args.eps)


if __name__ == '__main__':
    main()
