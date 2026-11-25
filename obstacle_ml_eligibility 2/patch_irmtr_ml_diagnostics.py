#!/usr/bin/env python3
"""Safely add coarse-eligibility *diagnostics only* to IRMTRMoreau.irmtr.

Designed against qiwang7777/Inexact_multilevel_TR/optimal_control/irmtr.py,
including local variations in whitespace and history-row formatting.
No mathematical conditions, Hessians, steps, or acceptance decisions are changed.

Usage, from the repository root:
    python3 obstacle_ml_eligibility/patch_irmtr_ml_diagnostics_v2.py --dry-run
    python3 obstacle_ml_eligibility/patch_irmtr_ml_diagnostics_v2.py
"""
from __future__ import annotations

import argparse
import ast
import re
import shutil
from pathlib import Path

INIT = re.compile(r"(?m)^(?P<indent>[ \t]*)reject_reason[ \t]*=[ \t]*(?:''|\"\")[ \t]*$")
TEST = re.compile(r"(?m)^(?P<indent>[ \t]*)eligible[ \t]*=[ \t]*ml_ok[ \t]+and[ \t]+et_ok[ \t]*$")
# In the public file, the history row ends 'rejection=reject_reason,step_ratio=...'.
# Match the *field*, not a line with a fixed count of leading spaces.
TRACE_FIELD = re.compile(r"\brejection[ \t]*=[ \t]*reject_reason\b")
MARKER = "zeta_moreau=zeta_moreau"

INIT_LINES = (
    "# Coarse-eligibility diagnostics: initialized on forced-fine steps too.",
    "ml_threshold = et_threshold = np.nan",
    "gf = gc = np.nan",
    "ml_ok = et_ok = False",
    "zeta_smooth = zeta_moreau = np.nan",
)
TEST_LINES = (
    "# Separate smooth and MY contributions to gradient inconsistency.",
    "zeta_smooth = self.norm(ell-1, cev.g - R @ ev.g)",
    "zeta_moreau = self.norm(",
    "    ell-1, (cev.G - cev.g) - R @ (ev.G - ev.g))",
)
TRACE_FIELDS = (
    "zeta_smooth=zeta_smooth,",
    "zeta_moreau=zeta_moreau,",
    "fine_gradient_norm=gf,",
    "coarse_gradient_norm=gc,",
    "ml_threshold=ml_threshold,",
    "et_threshold=et_threshold,",
    "ml_ok=ml_ok,et_ok=et_ok,",
    "force_fine=force_fine,",
)


def _unique(regex: re.Pattern, src: str, description: str) -> re.Match:
    found = list(regex.finditer(src))
    if len(found) != 1:
        raise ValueError(
            f"Expected exactly one {description}, found {len(found)}. "
            "No changes were made. Please share the relevant portion of "
            "optimal_control/irmtr.py instead of forcing a patch."
        )
    return found[0]


def _patch_source(source: str) -> str:
    if MARKER in source:
        raise ValueError("Coarse-eligibility diagnostics are already installed. Nothing changed.")
    init = _unique(INIT, source, "reject_reason initialization")
    test = _unique(TEST, source, "eligible = ml_ok and et_ok line")
    trace = _unique(TRACE_FIELD, source, "trace field rejection=reject_reason")
    start = source.rfind("self.history.append(", 0, trace.start())
    if start < 0 or trace.start() - start > 1200:
        raise ValueError(
            "Found rejection=reject_reason outside the expected self.history.append "
            "trace block. No changes were made."
        )
    # Insert from the bottom up, so offsets from the original text stay valid.
    operations = []

    def insert_after_line(match: re.Match, new_lines: tuple[str, ...]) -> None:
        i = source.find("\n", match.end())
        if i < 0:
            raise ValueError("Unexpected EOF after source anchor. No changes made.")
        indent = match.group("indent")
        payload = "".join(indent + line + "\n" for line in new_lines)
        operations.append((i + 1, payload))

    insert_after_line(init, INIT_LINES)
    insert_after_line(test, TEST_LINES)
    line_start = source.rfind("\n", 0, trace.start()) + 1
    indent = re.match(r"[ \t]*", source[line_start:]).group(0)
    # We insert immediately before a keyword within the dict(...) call. This
    # also works if rejection=reject_reason shares a line with another field.
    trace_payload = "".join((field + "\n" + indent) for field in TRACE_FIELDS)
    operations.append((trace.start(), trace_payload))

    updated = source
    for position, insertion in sorted(operations, reverse=True):
        updated = updated[:position] + insertion + updated[position:]
    ast.parse(updated)
    return updated


def patch(path: Path, *, dry_run: bool = False) -> None:
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    original = path.read_text(encoding="utf-8")
    modified = _patch_source(original)
    if dry_run:
        print(f"Dry-run successful: {path}")
        print("All three insertion points were found and the patched source parses.")
        print("No files were changed.")
        return
    backup = path.with_name(path.name + ".before_ml_diagnostics.bak")
    if backup.exists():
        raise ValueError(f"Backup already exists: {backup}. Refusing to overwrite it.")
    shutil.copy2(path, backup)
    path.write_text(modified, encoding="utf-8")
    print(f"Patched: {path}\nBackup: {backup}")
    print("Recorded coarse eligibility and gradient-mismatch diagnostics only.")
    print("IRMTR decisions, weighted transfers, and Bc=RBfP are unchanged.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("optimal_control/irmtr.py"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        patch(args.source, dry_run=args.dry_run)
    except (ValueError, SyntaxError) as exc:
        parser.exit(2, f"Patch not applied: {exc}\n")


if __name__ == "__main__":
    main()
