"""Side-by-side reports for the P1 dead-zone IRMTR/Moreau example.

No dependencies on the PDE solver: formatters can also be used for saved JSON.
"""
from __future__ import annotations

from pathlib import Path


_MISSING = object()


def _get(result, key):
    if isinstance(key, tuple):
        value = result
        for part in key:
            if not isinstance(value, dict):
                return _MISSING
            value = value.get(part, _MISSING)
            if value is _MISSING:
                return _MISSING
        return value
    return result.get(key, _MISSING)


def _format(value, kind):
    if value is _MISSING or value is None:
        return '—'
    if kind == 'bool':
        return 'Yes' if value else 'No'
    if kind == 'int':
        return f'{int(value):,d}'
    if kind == 'seconds':
        return f'{float(value):.3f}'
    if kind == 'objective':
        return f'{float(value):.9f}'
    if kind == 'scientific':
        return f'{float(value):.4e}'
    if kind == 'float':
        return f'{float(value):.3f}'
    raise ValueError(f'Unknown column format: {kind}')


# The data are pulled directly from each solver's summary (not recomputed).
# The prox upper bound is the certified stopping quantity, not just psi.
_METRICS = (
    ('Converged', 'converged', 'bool'),
    ('Moreau stages', 'stages', 'stage_count'),
    ('Total inner iterations', 'iterations', 'int'),
    ('CPU time (s)', 'elapsed', 'seconds'),
    ('Original objective J', 'original', 'objective'),
    ('Smooth part f', 'smooth', 'objective'),
    ('Dead-zone penalty phi', 'phi', 'objective'),
    ('Original prox residual', 'original_prox', 'scientific'),
    ('Gradient error at final z', 'prox_error', 'scientific'),
    ('Certified prox upper bound', 'prox_upper', 'scientific'),
    ('Coarse corrections attempted', 'coarse_attempted', 'int'),
    ('Coarse corrections accepted', 'accepted_coarse', 'int'),
    ('Rejected fine-level steps', 'rejected', 'int'),
    ('Gradient evaluations', 'gradient_evaluations', 'int'),
    ('State solves', ('pde_solve_counts', 'state_solves'), 'int'),
    ('Adjoint solves', ('pde_solve_counts', 'adjoint_solves'), 'int'),
    ('State CG iterations', ('pde_solve_counts', 'state_cg_iterations'), 'int'),
    ('Adjoint CG iterations', ('pde_solve_counts', 'adjoint_cg_iterations'), 'int'),
    ('Control L2 norm', 'z_L2', 'float'),
    ('Minimum control', 'z_min', 'float'),
    ('Maximum control', 'z_max', 'float'),
)


def _data_rows(results):
    if not results:
        raise ValueError('At least one solver result is required')
    columns = [r.get('method', r.get('key', 'Solver')) for r in results]
    rows = []
    for label, key, kind in _METRICS:
        entries = []
        for result in results:
            if kind == 'stage_count':
                value = result.get('stages', _MISSING)
                entries.append(_format(len(value) if isinstance(value, list) else _MISSING, 'int'))
            else:
                entries.append(_format(_get(result, key), kind))
        rows.append([label] + entries)
    return ['Metric'] + columns, rows


def comparison_table(results, *, markdown=False):
    """Format single/multilevel results for a terminal or Markdown report."""
    headings, rows = _data_rows(results)
    widths = [max(len(row[i]) for row in [headings, *rows])
              for i in range(len(headings))]
    if markdown:
        def cells(row):
            return '| ' + ' | '.join(str(v).ljust(w) for v,w in zip(row,widths)) + ' |'
        output = [cells(headings),
                  '| ' + ' | '.join('-'*w for w in widths) + ' |']
    else:
        def cells(row):
            return ' | '.join(str(v).ljust(w) for v,w in zip(row,widths))
        output = [cells(headings), '-+-'.join('-'*w for w in widths)]
    output.extend(cells(row) for row in rows)
    return '\n'.join(output)


def write_comparison_tables(results, output_dir):
    """Persist the same final table printed to stdout."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    txt = comparison_table(results)
    md = comparison_table(results, markdown=True)
    (output_dir / 'dead_zone_comparison.txt').write_text(
        'FINAL COMPARISON: IRMTR WITH MOREAU CONTINUATION\n'
        + txt + '\n', encoding='utf-8')
    (output_dir / 'dead_zone_comparison.md').write_text(
        '## Final comparison: IRMTR with Moreau continuation\n\n'
        + md + '\n', encoding='utf-8')
    return txt


def _continuation_iteration_rows(results):
    """Align per-Moreau-stage inner iteration counts across solver results.

    The continuation schedule is shared, but methods may converge after
    different numbers of outer stages. Missing stages are displayed as '—'.
    """
    if not results:
        raise ValueError('At least one solver result is required')
    methods = [str(r.get('method', r.get('key', 'Solver'))) for r in results]
    stage_lists = [r.get('stages', []) for r in results]
    max_stages = max((len(s) for s in stage_lists), default=0)
    rows = []
    for j in range(max_stages):
        at_stage = [stages[j] if j < len(stages) else None for stages in stage_lists]
        reference = next((s for s in at_stage if s is not None), None)
        mu = reference.get('mu') if reference else None
        tau = reference.get('tau') if reference else None
        values = []
        for stage in at_stage:
            if stage is None:
                values.append(None)
            elif mu is not None and abs(float(stage['mu']) - float(mu)) > 1e-10 * max(1.0, abs(float(mu))):
                raise ValueError('Incompatible mu schedules at continuation stage {}'.format(j))
            else:
                values.append(int(stage['inner_iterations']))
        rows.append((mu, tau, values))
    return methods, rows, [sum(int(s['inner_iterations']) for s in stages)
                            for stages in stage_lists]


def continuation_iterations_table(results, *, markdown=False):
    """Print one row per mu and a final sum, matching the semilinear report."""
    methods, rows, totals = _continuation_iteration_rows(results)
    headings = ['mu', 'inner tolerance tau'] + methods
    formatted = [[f'{mu:.4e}' if mu is not None else '—',
                  f'{tau:.4e}' if tau is not None else '—',
                  *[str(v) if v is not None else '—' for v in iters]]
                 for mu, tau, iters in rows]
    formatted.append(['TOTAL', '—', *[str(total) for total in totals]])
    widths = [max(len(str(row[i])) for row in [headings, *formatted])
              for i in range(len(headings))]
    if markdown:
        def line(row):
            return '| ' + ' | '.join(str(x).ljust(w) for x, w in zip(row, widths)) + ' |'
        output = [line(headings), '| ' + ' | '.join('-' * w for w in widths) + ' |']
    else:
        def line(row):
            return ' | '.join(str(x).ljust(w) for x, w in zip(row, widths))
        output = [line(headings), '-+-'.join('-' * w for w in widths)]
    output.extend(line(row) for row in formatted)
    return '\n'.join(output)


def write_continuation_iteration_tables(results, output_dir):
    """Save the per-mu iteration table as plain text, Markdown, and CSV."""
    import csv
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    heading = 'ITERATIONS BY MOREAU PARAMETER (INNER IRMTR / TR STEPS)'
    txt = continuation_iterations_table(results)
    (output_dir / 'dead_zone_iterations_by_mu.txt').write_text(
        heading + '\n' + txt + '\n', encoding='utf-8')
    (output_dir / 'dead_zone_iterations_by_mu.md').write_text(
        '## Iterations by Moreau continuation parameter\n\n'
        + continuation_iterations_table(results, markdown=True) + '\n', encoding='utf-8')
    methods, rows, totals = _continuation_iteration_rows(results)
    with (output_dir / 'dead_zone_iterations_by_mu.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['mu', 'tau', *methods])
        for mu, tau, values in rows:
            writer.writerow([mu, tau, *['' if v is None else v for v in values]])
        writer.writerow(['TOTAL', '', *totals])
    return txt
