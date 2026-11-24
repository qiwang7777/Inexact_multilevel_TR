"""Report plots for the dead-zone IRMTR experiment (same categories as semilinear)."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def make_plots(data, output_dir, *, source_dir=None, show=False):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir if source_dir is None else Path(source_dir)
    results = data['results']
    if not results:
        return

    def finish(name):
        plt.tight_layout()
        plt.savefig(output_dir/name, dpi=175, bbox_inches='tight')
        if not show:
            plt.close()

    # (1) IRMTR fine-level convergence, discontinuities between MY stages kept.
    plt.figure(figsize=(8.6,5.2))
    for result in results:
        max_level = (len(data['mesh_ns'])-1 if result['key']=='multi' else 0)
        trace = [row for row in result['trace'] if row.get('level') == max_level]
        y = np.array([row['certified_residual'] for row in trace
                      if row.get('certified_residual') is not None],dtype=float)
        if len(y):
            plt.semilogy(np.arange(1,len(y)+1),np.maximum(y,1e-16),
                         label=result['method'])
    plt.axhline(data['eps'],linestyle='--',label='original stopping tolerance')
    plt.xlabel('Fine-level trust-region steps (all MY stages)')
    plt.ylabel('Certified smoothed stationarity ||G_mu|| + e')
    plt.title('IRMTR versus single-level TR: convergence')
    plt.legend();plt.grid(alpha=.2)
    finish('dead_zone_convergence.png')

    # (2) Trust region model objective (not directly comparable across mu).
    plt.figure(figsize=(8.6,5.2))
    for result in results:
        max_level = len(data['mesh_ns'])-1 if result['key']=='multi' else 0
        trace = [row for row in result['trace'] if row.get('level')==max_level]
        y = np.array([row['F'] for row in trace if row.get('F') is not None], dtype=float)
        if len(y):
            plt.plot(np.arange(1,len(y)+1), y, label=result['method'])
    plt.xlabel('Fine-level trust-region steps (all MY stages)')
    plt.ylabel('Smoothed F_mu at iterate (mu changes across stages)')
    plt.title('IRMTR trust-region objective history')
    plt.legend();plt.grid(alpha=.2)
    finish('dead_zone_objective_history.png')

    # (3) Outer Moreau continuation / original nonsmooth stationarity.
    plt.figure(figsize=(8.6,5.2))
    for result in results:
        stages=result['stages']
        if stages:
            mu=np.array([row['mu'] for row in stages])
            upper=np.array([row['stationarity_upper'] for row in stages])
            plt.loglog(mu,np.maximum(upper,1e-16),marker='o',label=result['method'])
    plt.axhline(data['eps'],linestyle='--',label='eps')
    plt.xlabel('Moreau parameter mu')
    plt.ylabel('Original dead-zone proximal residual + certified gradient error')
    plt.title('Original nonsmooth stationarity by continuation stage')
    plt.gca().invert_xaxis();plt.legend();plt.grid(alpha=.2)
    finish('dead_zone_continuation.png')

    # (4) Iterations per continuation stage.
    plt.figure(figsize=(8.6,5.2))
    for result in results:
        rows=result['stages']
        if rows:
            plt.plot([q['j'] for q in rows], [q['inner_iterations'] for q in rows],
                     marker='o',label=result['method'])
    plt.xlabel('Moreau stage index')
    plt.ylabel('IRMTR / TR inner iterations')
    plt.title('Trust-region work per Moreau stage')
    plt.legend();plt.grid(alpha=.2)
    finish('dead_zone_work_by_mu.png')

    # (5) PDE/control fields for both solvers.
    saved={}
    for result in results:
        key=result['key']
        file=source_dir/f'dead_zone_{key}_solution.npz'
        if not file.exists():
            continue
        with np.load(file) as arr:
            saved[key]={k:arr[k].copy() for k in arr.files}
        s=saved[key]
        mesh=s['nodes'];tri=s['triangles']
        import matplotlib.tri as mtri
        t=mtri.Triangulation(mesh[:,0],mesh[:,1],tri)
        for field,label in [('z','Optimal P1 control'),('y','PDE state'),
                            ('desired','Desired state')]:
            plt.figure(figsize=(6,5.2))
            plt.tripcolor(t,s[field],shading='gouraud')
            plt.colorbar(label=label)
            plt.xlabel('$x_1$');plt.ylabel('$x_2$');plt.axis('equal')
            plt.title(f'{result["method"]}: {label}')
            finish(f'dead_zone_{key}_{field}.png')
        plt.figure(figsize=(6,5.2))
        plt.tripcolor(t,s['y']-s['desired'],shading='gouraud')
        plt.colorbar(label='$y_h-y_d$')
        plt.xlabel('$x_1$');plt.ylabel('$x_2$');plt.axis('equal')
        plt.title(f'{result["method"]}: tracking error')
        finish(f'dead_zone_{key}_state_error.png')

    if 'multi' in saved and 'single' in saved:
        a=saved['multi'];b=saved['single']
        import matplotlib.tri as mtri
        t=mtri.Triangulation(a['nodes'][:,0],a['nodes'][:,1],a['triangles'])
        plt.figure(figsize=(6,5.2))
        plt.tripcolor(t,a['z']-b['z'],shading='gouraud')
        plt.colorbar(label='Multilevel - Single-level')
        plt.xlabel('$x_1$');plt.ylabel('$x_2$');plt.axis('equal')
        plt.title('Final optimal-control difference')
        finish('dead_zone_control_difference.png')

    # (6) Stage-by-stage control norm, analogous to semilinear z-by-mu output.
    plt.figure(figsize=(8.6,5.2))
    for result in results:
        rows=result['stages']
        if rows:
            plt.semilogx([q['mu'] for q in rows],[q['z_L2'] for q in rows],
                         marker='o',label=result['method'])
    plt.xlabel('mu');plt.ylabel('||z||_{L2} (lumped mass)')
    plt.gca().invert_xaxis()
    plt.title('Control magnitude during MY continuation')
    plt.legend();plt.grid(alpha=.2)
    finish('dead_zone_control_by_mu.png')

    # Per-mu files are saved as compressed NPZ for complete P1 controls.
    # (7) Spatial control snapshots at each mu for each method.
    # The per-stage files are named and saved by run_continuation.
    for result in results:
        key=result['key']
        if key not in saved:
            continue
        snapshots=sorted(source_dir.glob(f'{key}_stage_*_mu_*.npz'))
        if not snapshots:
            continue
        arr=saved[key]
        import matplotlib.tri as mtri
        t=mtri.Triangulation(arr['nodes'][:,0],arr['nodes'][:,1],arr['triangles'])
        fig,axes=plt.subplots(2, int(np.ceil(len(snapshots)/2)),
                              figsize=(3.3*int(np.ceil(len(snapshots)/2)),6.5),
                              squeeze=False)
        panels=[]
        for name in snapshots:
            with np.load(name) as snap:
                panels.append((float(snap['mu']),snap['z'].copy()))
        vlim=max(max(np.max(np.abs(z)) for _,z in panels),1e-10)
        for ax,(mu,z) in zip(axes.ravel(),panels):
            picture=ax.tripcolor(t,z,shading='gouraud',vmin=-vlim,vmax=vlim)
            ax.set_title(f'mu={mu:g}');ax.set_aspect('equal');ax.set_xlabel('$x_1$');ax.set_ylabel('$x_2$')
        for ax in axes.ravel()[len(panels):]:
            ax.axis('off')
        fig.colorbar(picture, ax=axes.ravel().tolist(), label='P1 control z', shrink=.8)
        fig.suptitle(f'{result["method"]}: stage-by-stage optimal controls')
        fig.savefig(output_dir/f'dead_zone_{key}_controls_by_mu.png',dpi=160,bbox_inches='tight')
        if not show:
            plt.close(fig)

    if show:
        plt.show()
