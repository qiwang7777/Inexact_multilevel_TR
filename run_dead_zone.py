"""CLI example: python optimal_control/dead_zone/run_dead_zone.py --plot"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from dead_zone import Hierarchy, Parameters, solve_nested


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 optimal control with dead-zone penalty")
    parser.add_argument("--resolutions", default="32,64,128", help="coarse-to-fine n (default 32,64,128)")
    parser.add_argument("--maxiter", type=int, default=300, help="max iterations per grid")
    parser.add_argument("--tol", type=float, default=1e-7, help="absolute W-gradient-mapping tolerance")
    parser.add_argument("--output", default="dead_zone_results.npz", help="saved NumPy results")
    parser.add_argument("--plot", action="store_true", help="also save a PNG of finest state and control")
    args = parser.parse_args()
    ns = tuple(int(n) for n in args.resolutions.split(','))
    hierarchy = Hierarchy(ns, Parameters())
    print("Lumped-mass dead-zone optimal control (FISTA reference solver)")
    results = solve_nested(hierarchy, maxiter=args.maxiter, rtol=args.tol)
    problem, result = hierarchy.levels[-1], results[-1]
    y = problem.state(result.x)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {"x": problem.vertices, "triangles": problem.triangles,
            "control": result.x, "state": y, "desired_state": problem.yd,
            "mass_weights": problem.w, "objective": result.objective,
            "stationarity": result.stationarity,
            "levels": np.array(ns, dtype=int),
            "control_dofs": np.array([p.N for p in hierarchy.levels]),
            "iterations": np.array([r.iterations for r in results])}
    for ell, r in enumerate(results):
        data[f"history_{ell}"] = np.asarray(r.history).reshape(-1, 3)
    np.savez_compressed(output, **data)
    print(f"Saved {output}")
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        for ax, values, title in zip(axes,
                                     [result.x, y, problem.yd],
                                     ["Optimal control", "Computed state", "Desired state"]):
            pic = ax.tripcolor(problem.vertices[:,0], problem.vertices[:,1],
                               problem.triangles, values, shading='gouraud')
            ax.set_aspect('equal')
            ax.set_title(title)
            fig.colorbar(pic, ax=ax, shrink=.85)
        png = output.with_suffix('.png')
        fig.savefig(png, dpi=160)
        plt.close(fig)
        print(f"Saved {png}")


if __name__ == "__main__":
    main()
