"""Plots for P1 obstacle IRMTR/Moreau benchmark; supports --replot-data."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def make_plots(data, out, *, source_dir=None, show=False):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    source=out if source_dir is None else Path(source_dir)
    results=data['results']
    if not results:
        return
    def save(name):
        plt.tight_layout()
        plt.savefig(out/name,dpi=160,bbox_inches='tight')
        if not show:
            plt.close()
    plt.figure(figsize=(8,4.8))
    for result in results:
        maxlevel=0 if result['key']=='single' else len(data['mesh_ns'])-1
        rows=[r for r in result['trace'] if r['level']==maxlevel]
        if rows:
            y=np.array([r['certified_residual'] for r in rows],float)
            plt.semilogy(np.arange(1,len(y)+1),np.maximum(y,1e-15),
                         label=result['method'])
    plt.xlabel('Finest-level trust-region iterations')
    plt.ylabel('Smoothed gradient norm ||G_mu||_W (mu changes by stage)')
    plt.title('Single-level versus multilevel: inner convergence')
    plt.grid(alpha=.2);plt.legend()
    save('obstacle_convergence.png')
    plt.figure(figsize=(8,4.8))
    for result in results:
        st=result['stages']
        if st:
            plt.loglog([r['mu'] for r in st],
                       np.maximum([r['stationarity_upper'] for r in st],1e-15),
                       marker='o',label=result['method'])
    plt.axhline(data['eps'],linestyle='--',color='gray',label='target eps')
    plt.gca().invert_xaxis()
    plt.xlabel('Moreau parameter mu')
    plt.ylabel('Projected-gradient residual at feasible control')
    plt.title('Original constrained stationarity by Moreau stage')
    plt.grid(alpha=.2);plt.legend()
    save('obstacle_moreau_continuation.png')
    plt.figure(figsize=(8,4.8))
    for result in results:
        st=result['stages']
        if st:
            plt.plot([r['mu'] for r in st],[r['inner_iterations'] for r in st],
                     marker='o',label=result['method'])
    plt.gca().invert_xaxis()
    plt.xscale('log')
    plt.xlabel('Moreau parameter mu')
    plt.ylabel('Inner iterations')
    plt.title('Trust-region work by continuation stage')
    plt.grid(alpha=.2);plt.legend()
    save('obstacle_iterations_by_mu.png')
    plt.figure(figsize=(8,4.8))
    for result in results:
        maxlevel=0 if result['key']=='single' else len(data['mesh_ns'])-1
        rows=[r for r in result['trace'] if r['level']==maxlevel]
        if rows:
            plt.semilogy(np.arange(1,len(rows)+1),
                         np.maximum([r['delta'] for r in rows],1e-15),
                         label=result['method'])
    plt.xlabel('Finest-level iteration')
    plt.ylabel('Trust-region radius')
    plt.title('Trust-region radius history')
    plt.grid(alpha=.2);plt.legend()
    save('obstacle_radius_history.png')
    solutions={}
    for result in results:
        key=result['key']
        path=source/f'obstacle_{key}_solution.npz'
        if not path.is_file():
            continue
        with np.load(path) as arr:
            s={k:arr[k].copy() for k in arr.files}
        solutions[key]=s
        triang=mtri.Triangulation(s['nodes'][:,0],s['nodes'][:,1],s['triangles'])
        for field,label in [('z','Feasible obstacle solution'),('psi','Obstacle'),
                            ('forcing','Forcing g(x)'),('diffusion','Diffusion a(x)')]:
            if field not in s:
                continue
            plt.figure(figsize=(6,5))
            plt.tripcolor(triang,s[field],shading='gouraud')
            plt.colorbar(label=label)
            plt.axis('equal');plt.xlabel('$x_1$');plt.ylabel('$x_2$')
            plt.title(result['method']+': '+label)
            save(f'obstacle_{key}_{field}.png')
        plt.figure(figsize=(6,5))
        plt.tripcolor(triang,np.isclose(s['z'],s['psi'],atol=1e-6).astype(float),
                      shading='flat',vmin=0,vmax=1)
        plt.colorbar(label='Contact indicator (nodal approximate)')
        plt.axis('equal');plt.xlabel('$x_1$');plt.ylabel('$x_2$')
        plt.title(result['method']+': contact set')
        save(f'obstacle_{key}_contact.png')
    if 'single' in solutions and 'multi' in solutions:
        s=solutions['single'];m=solutions['multi']
        if np.array_equal(s['nodes'],m['nodes']):
            triang=mtri.Triangulation(m['nodes'][:,0],m['nodes'][:,1],m['triangles'])
            plt.figure(figsize=(6,5))
            plt.tripcolor(triang,m['z']-s['z'],shading='gouraud')
            plt.colorbar(label='Multilevel minus single-level')
            plt.axis('equal');plt.xlabel('$x_1$');plt.ylabel('$x_2$')
            plt.title('Final feasible solution difference')
            save('obstacle_solution_difference.png')
    if show:
        plt.show()
