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
    print(f'Original residual + certified error: {stationarity:.9e}')
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




# ---- Example 2: P1 elliptic control report (same module as semilinear) ----
import csv
from .pde.elliptic import smooth_value
from .regularization import moreau_value

def elliptic_original_values(m,z,mu,alpha,beta,pde):
    ell=len(pde.meshes)-1
    y,_=pde.state(next(i for i,q in enumerate(pde.meshes) if q.n==m.n),z)
    smooth=smooth_value(m,z,y,alpha)
    smoothed=smooth+moreau_value(m,z,mu,beta)
    violation=float(max(0,np.max(np.abs(z)-25)))
    original=(smooth+beta*np.dot(m.W,np.abs(z))) if violation<=1e-12 else np.inf
    zp=np.clip(z,-25,25)
    yp,_=pde.state(next(i for i,q in enumerate(pde.meshes) if q.n==m.n),zp)
    projected=smooth_value(m,zp,yp,alpha)+beta*np.dot(m.W,np.abs(zp))
    return dict(smoothed=float(smoothed),
                original=(float(original) if np.isfinite(original) else None),
                projected=float(projected),smooth=float(smooth),
                max_box_violation=violation)

def report_elliptic_results(rows, outdir):
    """Export every quantity needed for the publication comparison table.

    Reads the stored continuation stages and finest-level iteration trace;
    works both immediately after a solve and using --replot-data, without
    rerunning the PDE problem. Never confuses F(z) with F(clip(z)).
    """
    if not rows:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    by_name={r['method']:r for r in rows}
    names=['Single-level TR','Multilevel IRMTR']
    available=[name for name in names if name in by_name]
    all_mus=sorted({float(st['mu']) for r in rows for st in r['stages']}, reverse=True)
    stage_records=[]
    for r in rows:
        for st in r['stages']:
            tr=[t for t in r.get('trace', []) if
                t['level']==r['finest'] and np.isclose(t['mu'],st['mu'],rtol=0,atol=1e-12)]
            stage_records.append(dict(
                method=r['method'], stage=st['j'], mu=st['mu'], tau=st['tau'],
                inner_iterations=st['inner_iterations'],
                inner_converged=st['inner_converged'],
                inner_final_certified_residual=st['inner_final_residual'],
                stage_cpu_seconds=st['elapsed'],
                finest_coarse_candidates=sum(t['mode']=='coarse' for t in tr),
                finest_coarse_accepted=sum(t['mode']=='coarse' and t['accepted'] for t in tr),
                rejected_finest_steps=sum(not t['accepted'] for t in tr),
                proximal_residual=st.get('stationarity_hat'),
                proximal_error_bound=st.get('stationarity_error'),
                certified_proximal_residual=st.get('stationarity_upper'),
                feasible_proximal_residual=st.get('feasible_psi'),
                feasible_certified_residual=st.get('feasible_stationarity_upper'),
            ))
    with (outdir/'example2_stage_summary.csv').open('w',newline='') as f:
        cols=list(stage_records[0])
        wr=csv.DictWriter(f,fieldnames=cols)
        wr.writeheader();wr.writerows(stage_records)

    def cell(name, key, fmt):
        if name not in by_name:return '--'
        value=by_name[name].get(key)
        return '--' if value is None else format(value,fmt)

    def stage_count(name,mu):
        if name not in by_name:return '--'
        found=next((st for st in by_name[name]['stages']
                    if np.isclose(st['mu'],mu,rtol=0,atol=1e-12)),None)
        return str(found['inner_iterations']) if found else '--'

    lines=[('Quantity','Single-level TR','Multilevel IRMTR')]
    for mu in all_mus:
        lines.append((f'Inner iterations (mu={mu:g})',
                      stage_count(names[0],mu),stage_count(names[1],mu)))
    lines.extend([
        ('Total finest-level iterations',cell(names[0],'iterations','d'),cell(names[1],'iterations','d')),
        ('CPU time (s)',cell(names[0],'elapsed','.3f'),cell(names[1],'elapsed','.3f')),
        ('Finest-level coarse candidates',cell(names[0],'finest_coarse_candidates','d'),cell(names[1],'finest_coarse_candidates','d')),
        ('Finest-level coarse accepted',cell(names[0],'finest_coarse_accepted','d'),cell(names[1],'finest_coarse_accepted','d')),
        ('Rejected finest-level steps',cell(names[0],'rejected','d'),cell(names[1],'rejected','d')),
        ('Final original objective F(z)',cell(names[0],'original','.8f'),cell(names[1],'original','.8f')),
        ('Feasible objective F(clip(z))',cell(names[0],'projected','.8f'),cell(names[1],'projected','.8f')),
        ('Max box violation',cell(names[0],'max_box_violation','.4e'),cell(names[1],'max_box_violation','.4e')),
        ('Proximal residual psi_hat(z)',cell(names[0],'psi','.6e'),cell(names[1],'psi','.6e')),
        ('Proximal error bound e',cell(names[0],'psi_error','.6e'),cell(names[1],'psi_error','.6e')),
        ('Certified residual psi_hat(z)+e',cell(names[0],'psi_plus_e','.6e'),cell(names[1],'psi_plus_e','.6e')),
        ('Feasible proximal residual',cell(names[0],'feasible_psi','.6e'),cell(names[1],'feasible_psi','.6e')),
        ('Feasible stationarity upper bound',cell(names[0],'feasible_upper','.6e'),cell(names[1],'feasible_upper','.6e')),
        ('Converged',str(by_name[names[0]]['converged']) if names[0] in by_name else '--',
                     str(by_name[names[1]]['converged']) if names[1] in by_name else '--'),
    ])
    with (outdir/'example2_paper_summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerows(lines)

    labels={'Total finest-level iterations':r'Total finest-level iterations',
            'CPU time (s)':r'CPU time (s)',
            'Finest-level coarse candidates':r'Finest-level coarse candidates',
            'Finest-level coarse accepted':r'Accepted finest-level coarse steps',
            'Rejected finest-level steps':r'Rejected finest-level steps',
            'Final original objective F(z)':r'Final objective $F(z)$',
            'Feasible objective F(clip(z))':r'Feasible objective $F(\\Pi_{Z_{\\rm ad}}z)$',
            'Max box violation':r'Maximum box-constraint violation',
            'Proximal residual psi_hat(z)':r'Proximal residual $\\widehat{\\Psi}_r(z;g)$',
            'Proximal error bound e':r'Gradient error bound $e$',
            'Certified residual psi_hat(z)+e':r'Certified residual $\\widehat{\\Psi}_r(z;g)+e$',
            'Feasible proximal residual':r'Feasible proximal residual $\widehat{\Psi}_r(\Pi_{Z_{\rm ad}}z)$',
            'Feasible stationarity upper bound':r'Feasible upper bound $\widehat{\Psi}_r(\Pi_{Z_{\rm ad}}z)+e$',
            'Converged':r'Converged'}
    tex=[r'\\begin{table}[htbp]',r'    \\centering',
         r'    \\begin{tabular}{lcc}',r'\\toprule',
         r'\\textbf{Quantity} & \\textbf{Single-level} & \\textbf{Multilevel} \\\\',
         r'\\midrule']
    def texvalue(v):
        import re
        found=re.fullmatch(r'([+-]?[0-9]+(?:\.[0-9]+)?)[eE]([+-]?[0-9]+)', v)
        if found:
            return rf'${found.group(1)}\times10^{{{int(found.group(2))}}}$'
        return {'True':'Yes','False':'No'}.get(v,v)

    for i,(label,single,multi) in enumerate(lines[1:]):
        if label.startswith('Inner iterations (mu='):
            mu=label.removeprefix('Inner iterations (mu=').removesuffix(')')
            title=rf'Inner iterations ($\\mu={mu}$)'
        else:title=labels.get(label,label)
        # Numerical table values copied verbatim from recorded outputs.
        tex.append(f'    {title} & {texvalue(single)} & {texvalue(multi)} '+r'\\')
        if label in ('Total finest-level iterations','Rejected finest-level steps',
                     'Max box violation'):
            pass
        if label.startswith('Inner iterations (mu=') and (i+1==len(all_mus)):
            tex.append(r'\\midrule')
        if label=='Rejected finest-level steps':tex.append(r'\\midrule')
        if label=='Max box violation':tex.append(r'\\midrule')
    tex.extend([r'\\bottomrule',r'    \\end{tabular}',
                r'    \\caption{Comparison of single-level inexact TR and multilevel IRMTR for Example 2 under Moreau--Yosida continuation. A dash in the $F(z)$ row indicates a box-infeasible smoothed solution; the projected objective is reported separately. Both raw-iterate and projected-feasible control proximal residuals are reported separately.}',
                r'    \\label{tab:com_p1_linear}',r'\\end{table}'])
    # Backslashes were doubled above to keep editing source code easy;
    # emit ordinary LaTeX commands and line breaks to the .tex file.
    latex='\n'.join(tex).replace('\\\\','\\')
    # Restore the LaTeX row terminators (\) after simplifying commands.
    latex=latex.replace(' & '+str(lines[0][1])+' & ', ' & '+str(lines[0][1])+' & ')
    for label,a,b in lines[1:]:
        latex=latex.replace(f' & {a} & {b} '+chr(92)+'\n',f' & {a} & {b} '+chr(92)*2+'\n')
    latex=latex.replace(r'\\textbf{Multilevel} '+chr(92)+'\n',r'\\textbf{Multilevel} '+chr(92)*2+'\n')
    # When a value is rendered as inline TeX math, preserve the table row \\ terminator.
    latex='\n'.join(line+chr(92) if ' & ' in line and line.endswith(chr(92))
                    and not line.endswith(chr(92)*2) else line
                    for line in latex.splitlines())
    (outdir/'example2_paper_table.tex').write_text(latex+'\n')

    print('\n========== Example 2 publication comparison ==========')
    widths=[max(len(str(row[j])) for row in lines) for j in range(3)]
    for i,row in enumerate(lines):
        print(' | '.join(str(v).ljust(widths[j]) for j,v in enumerate(row)))
        if i==0:print('-+-'.join('-'*w for w in widths))
    print('NOTE: F(z) is undefined as a finite number at box-infeasible controls;')
    print('      F(clip(z)) is evaluated at a *different feasible control*.')
    print('      Raw and feasible-control proximal residuals are reported separately.')
    print('Saved: example2_stage_summary.csv, example2_paper_summary.csv,')
    print('       example2_paper_table.tex')

