"""CSV/NPZ control export and reproducible performance/result reports."""

from pathlib import Path
import numpy as np
from .geometry import element_geometry


def export_stage_controls(mesh, out, method_name, output_dir='optimal_z_by_mu'):
    """Export all P0 controls by MY stage, plus the final feasible control.

    Each CSV row corresponds to one triangle. z_raw is the smoothed iterate,
    z_feasible is its box projection, and z_optimal is the actually returned
    control. Files are written AFTER timing the solver, for a fair benchmark.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tag = 'single' if 'Single' in method_name else 'multilevel'
    centers = mesh.nodes[mesh.elements].mean(axis=1)
    ids = np.arange(len(mesh.elements))
    raw_values, feasible_values, mus, taus = [], [], [], []

    for stage in out.history:
        z_raw = np.asarray(stage['z_end'], dtype=float)
        z_feas = np.clip(z_raw, -25.0, 25.0)
        j, mu = stage['j'], stage['mu']
        data = np.column_stack((ids, centers, z_raw, z_feas))
        path = directory / f'{tag}_outer_j{j:02d}_mu{mu:.3e}.csv'
        np.savetxt(path, data, delimiter=',', comments='',
                   header='triangle_id,center_x,center_y,z_raw,z_feasible',
                   fmt=['%d', '%.12e', '%.12e', '%.16e', '%.16e'])
        raw_values.append(z_raw)
        feasible_values.append(z_feas)
        mus.append(mu)
        taus.append(stage['tau'])
        print(f'  Saved stage j={j}, mu={mu:.3e}: {path}')

    final_path = directory / f'{tag}_optimal_z.csv'
    final_data = np.column_stack((ids, centers, out.z))
    np.savetxt(final_path, final_data, delimiter=',', comments='',
               header='triangle_id,center_x,center_y,z_optimal',
               fmt=['%d', '%.12e', '%.12e', '%.16e'])

    archive = directory / f'{tag}_all_z_stages.npz'
    np.savez_compressed(archive,
                        mu=np.asarray(mus), tau=np.asarray(taus),
                        z_raw=np.stack(raw_values),
                        z_feasible=np.stack(feasible_values),
                        z_optimal=np.asarray(out.z),
                        element_centers=centers,
                        elements=mesh.elements,
                        nodes=mesh.nodes)
    print(f'  Final optimal z (all {len(out.z)} P0 values): {final_path}')
    print(f'  Stage-by-stage NumPy archive: {archive}')
    return final_path, archive


def report_solution(name, solver, out, wall, original_r, eps):
    """Report the actual nonsmooth objective and original proximal residual."""
    finest_level = len(solver.levels)-1
    mesh = solver.levels[-1].mesh
    z = out.z
    u = solver.problem.solve_state(mesh, z, tol=solver.cfg.state_tol)
    value_smooth = solver.problem.value_f(mesh, z, u, solver.cfg.alpha, solver.cfg.desired_state)
    areas, _ = element_geometry(mesh)
    box_violation = float(np.max(np.maximum(np.abs(z)-25.0, 0.0)))
    l1_term = solver.cfg.beta * np.dot(areas, np.abs(z))
    original_value = (value_smooth + l1_term if box_violation < 1e-12 else np.inf)
    # This is an independent final evaluation, not a claim of verified inexactness.
    tau_final = out.history[-1]["tau"]
    stationarity, psi, e = solver.original_stationarity(z, original_r, tau_final)
    histories = [row for row in solver.history if row['level'] == finest_level]
    accepted_coarse = sum(row['mode']=='coarse' and row['accepted'] for row in histories)
    rejections = sum(not row['accepted'] for row in histories)
    z_feas = np.clip(z, -25.0, 25.0)

    u_feas = solver.problem.solve_state(mesh, z_feas,tol=solver.cfg.state_tol)

    F_feas = solver.problem.value_f(mesh, z_feas, u_feas,solver.cfg.alpha,solver.cfg.desired_state)

    F_feas += solver.cfg.beta * np.dot(areas, np.abs(z_feas))

    tau_final = out.history[-1]["tau"]

    stat_feas, psi_feas, e_feas = solver.original_stationarity(z_feas, original_r, tau_final)

    print(f"Feasible original objective: {F_feas:.12e}")
    print(f"Projected-control proximal residual: {psi_feas:.9e}")
    print(f"Projected-control stationarity + e: {stat_feas:.9e}") 
    print(f'\n========== {name}: RESULT ==========')
    print('Outer converged:', out.converged)
    print(f'Elapsed wall time: {wall:.3f} seconds')
    print('Fine trust-region steps:', len(histories))
    print('Accepted coarse corrections:', accepted_coarse)
    print('Rejected fine steps:', rejections)
    print(f'Original proximal residual: {psi:.9e}')
    print(f'Original residual + prescribed error: {stationarity:.9e}')
    print(f'Box violation (max): {box_violation:.9e}')
    print(f'Original nonsmooth objective (inf if infeasible): {original_value:.12e}')
    print(f'Smooth f(z): {value_smooth:.12e}  L1 penalty: {l1_term:.12e}')
    print('Coarse eligibility / attempts / used / fallback:', solver.counters)
    print('\nStage | mu       | tau      | inner steps | inner ok | prox residual | reason')
    for stage in out.history:
        print(f"{stage['j']:5d} | {stage['mu']:.2e} | {stage['tau']:.2e} |"
              f" {stage['inner_iterations']:11d} | {str(stage['inner_converged']):>8s} |"
              f" {stage['feasible_psi']:.5e} | {stage['inner_reason']}")
    if not out.converged:
        print('WARNING: Full original-problem stopping criterion was not met.')
    return dict(name=name, time=wall, iterations=len(histories),
                converged=out.converged, prox=psi, original_F=original_value,
                box_violation=box_violation)


