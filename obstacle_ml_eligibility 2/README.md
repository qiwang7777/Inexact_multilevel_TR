# Coarse-eligibility diagnostic patch v2

The original patcher stopped with:

    Expected exactly one source anchor for trace row; found 0. Nothing changed.

Its history-recording anchor required an exact indentation/spacing pattern.
This v2 script matches the `rejection=reject_reason` field in
`self.history.append(...)` instead, allowing local formatting differences.
It is designed against the public 2026 `optimal_control/irmtr.py` at
https://github.com/qiwang7777/Inexact_multilevel_TR/blob/main/optimal_control/irmtr.py .

## Install

Extract the ZIP to the root of your local repository, replacing the earlier
`obstacle_ml_eligibility/patch_irmtr_ml_diagnostics.py` script. From that root:

    python3 obstacle_ml_eligibility/patch_irmtr_ml_diagnostics.py --dry-run
    python3 obstacle_ml_eligibility/patch_irmtr_ml_diagnostics.py
    python3 -m py_compile optimal_control/irmtr.py

The patcher makes a backup of `optimal_control/irmtr.py` at
`optimal_control/irmtr.py.before_ml_diagnostics.bak` and stops rather than
writing if it cannot locate exactly one occurrence of each anchor.

## Run the diagnostic

    python3 run_obstacle_irmtr.py \
      --solver multi --base-n 64 \
      --mu0 0.2 --tau0 1e-3 --theta-mu 0.5 --theta-tau 0.75 \
      --eps 1e-4 --stop-on raw --warm-start unchanged \
      --cg-steps 200 --max-inner 2000 --max-outer 22 \
      --kappa-ml 0.25 --kappa-et 0.2 --fine-every 5 \
      --plot-dir obstacle_ml_diag_baseline

    python3 obstacle_ml_eligibility/analyze_coarse_eligibility.py \
      obstacle_ml_diag_baseline/obstacle_multi_trace.csv

The code only records the individual ML / ET gates and smooth/Moreau
contributions to the coarse gradient mismatch. It does not change
the eligibility decisions, trust-region steps, R/P transfers, or Galerkin
Hessian Bc=R Bf P.

To restore the original solver:

    cp optimal_control/irmtr.py.before_ml_diagnostics.bak optimal_control/irmtr.py

A syntax-valid synthetic-source test was run for three trace-formatting
variants and verified both dry-run and backup behavior. The patch has not
been run against your uncommitted local solver copy or a complete PDE benchmark.
