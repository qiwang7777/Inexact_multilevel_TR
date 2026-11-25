#!/usr/bin/env python3
"""Summarize why coarse corrections fail eligibility in IRMTR trace CSV.

Input: obstacle_multi_trace.csv, after applying the *diagnostic-only* patch.
Prints totals for finest-level iterations and per-Moreau-stage diagnostics.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import math
from statistics import median


def get_float(row, name):
    try:
        return float(row.get(name, 'nan'))
    except (ValueError, TypeError):
        return float('nan')


def get_bool(row, name):
    return str(row.get(name, '')).strip().lower() in ('1', 'true', 'yes')


def pct(vals, q):
    vals = sorted(x for x in vals if math.isfinite(x))
    if not vals:
        return float('nan')
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = math.floor(pos)
    hi = min(len(vals)-1, lo + 1)
    return vals[lo] + (vals[hi]-vals[lo])*(pos-lo)


def summarize(rows):
    n=len(rows)
    checked=[r for r in rows if math.isfinite(get_float(r,'ml_threshold'))]
    forced=[r for r in rows if get_bool(r,'force_fine')]
    fails_ml=[r for r in checked if not get_bool(r,'ml_ok')]
    fails_et=[r for r in checked if not get_bool(r,'et_ok')]
    eligible=[r for r in checked if get_bool(r,'eligible')]
    coarse_used=[r for r in checked if r.get('mode','').strip().lower() == 'coarse']
    coarse_accepted=[r for r in coarse_used if get_bool(r,'accepted')]
    ratio=[get_float(r,'zeta')/max(get_float(r,'fine_gradient_norm'),1e-300)
           for r in checked]
    mismatch_smooth=[get_float(r,'zeta_smooth') for r in checked]
    mismatch_moreau=[get_float(r,'zeta_moreau') for r in checked]
    return dict(iterations=n,forced=len(forced),checked=len(checked),
                failed_ml=len(fails_ml),failed_et=len(fails_et),
                failed_both=sum(not get_bool(r,'ml_ok') and not get_bool(r,'et_ok')
                                for r in checked),
                eligible=len(eligible),used=len(coarse_used),accepted=len(coarse_accepted),
                median_zeta_over_gf=pct(ratio,0.5),
                p90_zeta_over_gf=pct(ratio,0.9),
                median_zeta_smooth=pct(mismatch_smooth,0.5),
                median_zeta_moreau=pct(mismatch_moreau,0.5))


def analyze(csvpath):
    with csvpath.open('r',newline='',encoding='utf-8') as fh:
        rows=list(csv.DictReader(fh))
    if not rows:
        raise SystemExit('Empty trace CSV')
    needed=['ml_ok','et_ok','zeta_smooth','zeta_moreau','fine_gradient_norm']
    missing=[name for name in needed if name not in rows[0]]
    if missing:
        raise SystemExit('Missing diagnostic fields: '+', '.join(missing)+'\n'
                         'Apply patch_irmtr_ml_diagnostics.py, rerun multilevel, and analyze the NEW trace.')
    # Finest level only; sum of all-level rows is not a count of fine TR steps.
    level=max(int(r['level']) for r in rows)
    fine=[r for r in rows if int(r['level']) == level]
    result=summarize(fine)
    print(f'Finest level: {level}; number of finest-level TR iterations: {len(fine)}')
    print('Coarse-eligibility diagnostics (some failure counts overlap):')
    for name in ('forced','checked','failed_ml','failed_et','failed_both',
                 'eligible','used','accepted','median_zeta_over_gf',
                 'p90_zeta_over_gf','median_zeta_smooth','median_zeta_moreau'):
        val=result[name]
        print(f'  {name:24s} {val:12.5g}' if isinstance(val,float)
              else f'  {name:24s} {val:12d}')
    print('\nPer-Moreau-stage eligibility:')
    groups=defaultdict(list)
    for r in fine:
        groups[get_float(r,'mu')].append(r)
    head=('mu','steps','checked','ml_fail','et_fail','eligible','used','accepted',
          'med_zeta/gf','med_z_smooth','med_z_moreau')
    fmts=('{:.4e}','{:d}','{:d}','{:d}','{:d}','{:d}','{:d}','{:d}',
          '{:.2e}','{:.2e}','{:.2e}')
    widths=(13,7,8,8,8,9,6,9,13,14,14)
    def row_format(values):
        return '  '.join(str(v).rjust(w) for v,w in zip(values,widths))
    print(row_format(head))
    for mu in sorted(groups,reverse=True):
        s=summarize(groups[mu])
        vals=(mu,s['iterations'],s['checked'],s['failed_ml'],s['failed_et'],
              s['eligible'],s['used'],s['accepted'],s['median_zeta_over_gf'],
              s['median_zeta_smooth'],s['median_zeta_moreau'])
        print(row_format([fmt.format(v) for fmt,v in zip(fmts, vals)]))
    print('\nInterpretation:')
    if result['checked'] == 0:
        print('  No coarse eligibility checks were recorded. Inspect fine_every and solver setup.')
    else:
        if result['failed_ml'] > result['checked'] * .8:
            print('  ML consistency test fails on >80% of checked iterations.')
            print('  Tuning kappa_et or increasing coarse iterations will not fix this gate.')
        if result['failed_et'] > result['checked'] * .8:
            print('  Coarse gradient-size test fails on >80% of checked iterations.')
            print('  Diagnose coarse gradient magnitude before changing kappa_ml.')
        if result['median_zeta_moreau'] > 2*result['median_zeta_smooth']:
            print('  The Moreau-gradient inconsistency dominates typical smooth inconsistency.')
            print('  This points to a coarse active-set/projection mismatch, especially at small mu.')
        elif result['median_zeta_smooth'] > 2*result['median_zeta_moreau']:
            print('  The smooth gradient inconsistency dominates typical Moreau inconsistency.')
        if result['eligible'] > 0 and result['used'] < result['eligible']:
            print('  Some eligible coarse solves were rejected by later model/step tests.')
    print('  Compare CPU time and true fine-level iterations; more eligibility alone is not success.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace',type=Path,help='path to obstacle_multi_trace.csv')
    args=p.parse_args()
    analyze(args.trace)

if __name__=='__main__':
    main()
