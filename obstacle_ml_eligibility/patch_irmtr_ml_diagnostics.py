#!/usr/bin/env python3
"""Add non-invasive coarse-eligibility diagnostics to a local irmtr.py.

No changes to eligibility, step choice, Hessians, or trust-region behavior.
Backs up the original file before the first modification. Works with the
source anchors present in the public Nov 2026 repository version.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import shutil

INIT_OLD = "            reject_reason = ''\n"
INIT_NEW = (INIT_OLD +
"            # Diagnostic values also exist on forced-fine iterations.\n"
"            ml_threshold = et_threshold = np.nan\n"
"            gf = gc = np.nan\n"
"            ml_ok = et_ok = False\n"
"            zeta_smooth = zeta_moreau = np.nan\n")

TEST_OLD = "                eligible = ml_ok and et_ok\n"
TEST_NEW = (TEST_OLD +
"                # Decompose the coarse/fine gradient inconsistency.\n"
"                # Exact in obstacle backend; no additional PDE solves.\n"
"                zeta_smooth = self.norm(ell-1, cev.g - R @ ev.g)\n"
"                zeta_moreau = self.norm(\n"
"                    ell-1, (cev.G - cev.g) - R @ (ev.G - ev.g))\n")

TRACE_OLD = "                                      zeta=zeta,eligible=eligible,\n"
TRACE_NEW = (TRACE_OLD +
"                                      zeta_smooth=zeta_smooth,\n"
"                                      zeta_moreau=zeta_moreau,\n"
"                                      fine_gradient_norm=gf,\n"
"                                      coarse_gradient_norm=gc,\n"
"                                      ml_threshold=ml_threshold,\n"
"                                      et_threshold=et_threshold,\n"
"                                      ml_ok=ml_ok,et_ok=et_ok,\n"
"                                      force_fine=force_fine,\n")


def patch(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f'File not found: {path}')
    original = path.read_text(encoding='utf-8')
    if 'zeta_moreau=zeta_moreau' in original:
        raise SystemExit('Diagnostics already present. Nothing changed.')
    text = original
    for label, old, new in [
        ('initialization', INIT_OLD, INIT_NEW),
        ('eligibility test', TEST_OLD, TEST_NEW),
        ('trace row', TRACE_OLD, TRACE_NEW),
    ]:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f'Expected exactly one source anchor for {label}; found {count}. Nothing changed.')
        text = text.replace(old, new, 1)
    try:
        ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        raise SystemExit(f'Patched code did not parse: {exc}. Nothing changed.')
    backup = path.with_name(path.name + '.before_ml_diagnostics.bak')
    if backup.exists():
        raise SystemExit(f'Backup exists already: {backup}. Refusing to overwrite it.')
    shutil.copy2(path, backup)
    path.write_text(text, encoding='utf-8')
    print('Patched:', path)
    print('Backup: ', backup)
    print('No mathematical or solver decisions were changed.')
    print('Added: smooth/Moreau mismatch, eligibility thresholds and flags per trace row.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('optimal_control/irmtr.py'))
    args=p.parse_args()
    patch(args.source)

if __name__=='__main__':
    main()
