"""Publication-oriented plots of P0 controls and continuation history."""

import numpy as np
from collections import Counter
from .geometry import element_geometry


def plot_results(mesh, solver, out, eps=1e-3):
    """Display results in Spyder Plots pane; do not save files."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from collections import Counter
    triang = mtri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh.elements)
    u = solver.problem.solve_state(mesh, out.z, tol=solver.cfg.state_tol)
    fig, axs = plt.subplots(1,2,figsize=(11,4.8))
    a=axs[0].tripcolor(triang,facecolors=out.z,shading='flat',cmap='coolwarm')
    fig.colorbar(a,ax=axs[0],label='Control z')
    axs[0].set_title('Finest-level control')
    b=axs[1].tripcolor(triang,u,shading='gouraud',cmap='viridis')
    fig.colorbar(b,ax=axs[1],label='State u')
    axs[1].set_title('Finest-level PDE state')
    for ax in axs: ax.set_aspect('equal');ax.set_xlabel('x');ax.set_ylabel('y')
    fig.tight_layout()

    hist=[r for r in solver.history if r['level']==len(solver.levels)-1]
    if hist:
        fig2,axes=plt.subplots(2,2,figsize=(11,7))
        x=np.arange(len(hist))
        axes[0,0].plot(x,[r['F'] for r in hist]);axes[0,0].set_title('Finest-level smoothed objective')
        axes[0,1].semilogy(x,[max(r['residual'],1e-16) for r in hist]);axes[0,1].set_title('Finest-level stationarity')
        axes[1,0].plot(x,[r['rho'] for r in hist]);axes[1,0].set_title('Acceptance ratio')
        axes[1,1].semilogy(x,[max(r['delta'],1e-16) for r in hist]);axes[1,1].set_title('Trust-region radius')
        for ax in axes.flat:ax.grid(True,alpha=.25);ax.set_xlabel('Finest-level iteration')
        fig2.tight_layout()
    if out.history:
        fig3,ax=plt.subplots(figsize=(6,4))
        ax.semilogy([r['j'] for r in out.history],
                    [r['feasible_stationarity_upper'] for r in out.history],'o-')
        ax.axhline(eps,color='gray',linestyle='--',label='epsilon')
        ax.set_title('Feasible proximal stationarity + e');ax.set_xlabel('Smoothing stage')
        ax.grid(True,alpha=.25);ax.legend();fig3.tight_layout()
    counts=Counter((r['level'],r['mode']) for r in solver.history)
    print('TR steps by level and mode:',dict(counts))
    plt.show()


def plot_final_controls_vs_reference(
        finest_mesh, out_single, out_multi, z_reference,
        savepath="final_controls_vs_reference.png"):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    coords = np.asarray(finest_mesh.nodes)
    triangles = np.asarray(finest_mesh.elements)

    triang = mtri.Triangulation(
        coords[:, 0],
        coords[:, 1],
        triangles
    )

    # Final feasible control from each method
    z_single = np.clip(out_single.z, -25.0, 25.0)
    z_multi = np.clip(out_multi.z, -25.0, 25.0)

    # Reference solution on the same P0 mesh
    z_ref = np.asarray(z_reference, dtype=float)

    if not (
        len(z_single) == len(z_multi)
        == len(z_ref) == len(triangles)
    ):
        raise ValueError(
            "All controls must use the same P0 mesh."
        )

    # Correct discrete L2 norm
    areas, _ = element_geometry(finest_mesh)

    error_single = np.sqrt(
        np.dot(areas, (z_single - z_ref)**2)
    )
    error_multi = np.sqrt(
        np.dot(areas, (z_multi - z_ref)**2)
    )

    fig, axes = plt.subplots(
        1, 3,
        figsize=(13, 4),
        constrained_layout=True
    )

    controls = [
        z_ref,
        z_single,
        z_multi
    ]

    titles = [
        "(a) Reference solution",
        "(b) Single-level TR",
        "(c) Multilevel IRMTR"
    ]

    vmin = min(np.min(z) for z in controls)
    vmax = max(np.max(z) for z in controls)

    for ax, z, title in zip(axes, controls, titles):
        pc = ax.tripcolor(
            triang,
            facecolors=z,
            shading="flat",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    cbar = fig.colorbar(
        pc,
        ax=axes,
        shrink=0.85,
        pad=0.02
    )
    cbar.set_label("Control z")

    fig.savefig(
        savepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print("\n===== CONTROL ERRORS =====")
    print(f"Single-level L2 error: {error_single:.9e}")
    print(f"Multilevel L2 error:   {error_multi:.9e}")


def plot_z_by_mu_compare(finest_mesh, out_single, out_multi,
                         lb=-25.0, ub=25.0,
                         clip_to_box=True,
                         cmap='viridis',
                         diff_cmap='coolwarm'):
    """
    Plot the control z after every outer mu-stage.

    Rows: outer stages (matched by mu)
    Col 0: Single-level TR
    Col 1: Multilevel IRMTR
    Col 2: Difference = multi - single

    Assumes each entry in out_single.history / out_multi.history contains:
        'mu' and either 'z' or 'z_end'
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import TwoSlopeNorm

    # ---- mesh geometry ----
    coords = np.asarray(finest_mesh.nodes)       # shape (n_nodes, 2)
    tris   = np.asarray(finest_mesh.elements)    # shape (n_triangles, 3)

    x = coords[:, 0]
    y = coords[:, 1]
    triang = mtri.Triangulation(x, y, tris)

    def get_stage_z(rec):
        if 'z_end' in rec:
            z = np.asarray(rec['z_end']).copy()
        elif 'z' in rec:
            z = np.asarray(rec['z']).copy()
        else:
            raise KeyError("Each outer-history record must contain 'z_end' or 'z'.")
        if clip_to_box:
            z = np.clip(z, lb, ub)
        return z

    # ---- collect stages by mu ----
    mus_single = [rec['mu'] for rec in out_single.history]
    mus_multi  = [rec['mu'] for rec in out_multi.history]

    # only compare stages that exist in both histories
    mus = sorted(set(mus_single).intersection(set(mus_multi)), reverse=True)

    if len(mus) == 0:
        print("No common mu stages found between single and multilevel histories.")
        return

    single_by_mu = {rec['mu']: rec for rec in out_single.history}
    multi_by_mu  = {rec['mu']: rec for rec in out_multi.history}

    # ---- color scales ----
    all_controls = []
    all_diffs = []

    for mu in mus:
        z_single = get_stage_z(single_by_mu[mu])
        z_multi  = get_stage_z(multi_by_mu[mu])
        all_controls.extend([z_single, z_multi])
        all_diffs.append(z_multi - z_single)

    vmin = min(np.min(z) for z in all_controls)
    vmax = max(np.max(z) for z in all_controls)

    diff_abs = max(np.max(np.abs(d)) for d in all_diffs)
    if diff_abs == 0:
        diff_abs = 1e-14
    diff_norm = TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs)

    # ---- figure ----
    nrows = len(mus)
    fig, axes = plt.subplots(
        nrows, 3,
        figsize=(14, 3.6 * nrows),
        squeeze=False,
        constrained_layout=True
    )

    mappable_control = None
    mappable_diff = None

    for i, mu in enumerate(mus):
        rec_s = single_by_mu[mu]
        rec_m = multi_by_mu[mu]

        z_s = get_stage_z(rec_s)
        z_m = get_stage_z(rec_m)
        z_d = z_m - z_s

        # column 0: single
        ax = axes[i, 0]
        pc = ax.tripcolor(
            triang, facecolors=z_s, shading='flat',
            cmap=cmap, vmin=vmin, vmax=vmax
        )
        mappable_control = pc
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Single-level TR\n"
            f"j={rec_s.get('j', i)}, mu={mu:.3e}\n"
            f"min={np.min(z_s):.2f}, max={np.max(z_s):.2f}",
            fontsize=10
        )

        # column 1: multilevel
        ax = axes[i, 1]
        pc = ax.tripcolor(
            triang, facecolors=z_m, shading='flat',
            cmap=cmap, vmin=vmin, vmax=vmax
        )
        mappable_control = pc
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Multilevel IRMTR\n"
            f"j={rec_m.get('j', i)}, mu={mu:.3e}\n"
            f"min={np.min(z_m):.2f}, max={np.max(z_m):.2f}",
            fontsize=10
        )

        # column 2: difference
        ax = axes[i, 2]
        pd = ax.tripcolor(
            triang, facecolors=z_d, shading='flat',
            cmap=diff_cmap, norm=diff_norm
        )
        mappable_diff = pd
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Difference (multi - single)\n"
            f"max|diff|={np.max(np.abs(z_d)):.3e}\n"
            f"L2 diff={np.sqrt(np.dot(element_geometry(finest_mesh)[0], z_d*z_d)):.3e}",
            fontsize=10
        )

    # ---- colorbars ----
    cbar1 = fig.colorbar(
        mappable_control,
        ax=axes[:, :2].ravel().tolist(),
        shrink=0.95,
        pad=0.02
    )
    cbar1.set_label(r"Optimal control $z_{\mu_j}$")

    cbar2 = fig.colorbar(
        mappable_diff,
        ax=axes[:, 2].ravel().tolist(),
        shrink=0.95,
        pad=0.02
    )
    cbar2.set_label(r"$z_{\mu_j}^{\mathrm{multi}} - z_{\mu_j}^{\mathrm{single}}$")

    fig.suptitle("Control z after each outer μ-stage", fontsize=15)
    fig.savefig("optimal_z_all_mu_comparison.png", dpi=300, bbox_inches="tight")

    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        plt.show()

    plt.close(fig)


def plot_continuation_history_compare(out_single, out_multi, eps=None,
                                      savepath="continuation_history_compare.png"):
    """Compare inner TR iterations and feasible stationarity upper bound by Moreau stage."""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib

    def extract_stage_data(out):
        mus = []
        inner_iters = []
        prox_vals = []

        for rec in out.history:
            mus.append(rec['mu'])
            inner_iters.append(rec['inner_iterations'])

            #prox_vals.append(rec['psi'])
            prox_vals.append(rec['feasible_stationarity_upper'])

        return (np.asarray(mus, dtype=float),
                np.asarray(inner_iters, dtype=float),
                np.asarray(prox_vals, dtype=float))

    mu_s, it_s, prox_s = extract_stage_data(out_single)
    mu_m, it_m, prox_m = extract_stage_data(out_multi)

    # We plot against stage index so the two methods align cleanly,
    # and use mu_j as tick labels.
    nstage = max(len(mu_s), len(mu_m))
    x_s = np.arange(len(mu_s))
    x_m = np.arange(len(mu_m))

    fig, axes = plt.subplots(1, 2, figsize=(13,4.9), sharex=False,
                             constrained_layout=True)

    # ---- top: inner iterations ----
    ax = axes[0]
    ax.plot(x_s, it_s, marker='o', linewidth=2, label='Single-level TR')
    ax.plot(x_m, it_m, marker='s', linewidth=2, label='Multilevel IRMTR')
    ax.set_ylabel("Inner TR iterations")
    ax.set_title("Moreau continuation history")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right')

    # annotate values
    for x, y in zip(x_s, it_s):
        ax.annotate(f"{int(y)}", (x, y), textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=8)
    for x, y in zip(x_m, it_m):
        ax.annotate(f"{int(y)}", (x, y), textcoords="offset points",
                    xytext=(0, -12), ha='center', fontsize=8)

    # ---- bottom: proximal stationarity ----
    ax = axes[1]
    ax.semilogy(x_s, prox_s, marker='o', linewidth=2, label='Single-level TR')
    ax.semilogy(x_m, prox_m, marker='s', linewidth=2, label='Multilevel IRMTR')

    if eps is not None:
        ax.axhline(eps, color='k', linestyle='--', linewidth=1.5,
                   label=rf'$\varepsilon={eps:.1e}$')

    ax.set_ylabel(r"Feasible stationarity $\psi_r(z_{\mathrm{feas}})+e$")
    ax.set_xlabel(r"Outer stage $\mu_j$")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right')

    # use mu values as x tick labels
    # if both methods have same mu schedule, single labels are enough
    if len(mu_s) >= len(mu_m):
        ticks = x_s
        labels = [f"{mu:.2e}" for mu in mu_s]
    else:
        ticks = x_m
        labels = [f"{mu:.2e}" for mu in mu_m]

    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels(labels, rotation=30, ha='right')

    fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()

    plt.close(fig)


