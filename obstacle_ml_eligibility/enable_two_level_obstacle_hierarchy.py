#!/usr/bin/env python3
"""Optionally allow --mesh-ns 128,256 in the obstacle runner (default unchanged)."""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import shutil

ARG_ANCHOR = "    ap.add_argument('--base-n',type=int,default=64,help='3 meshes n,2n,4n; default 64 gives 4225,16641,66049 DOF')\n"
ARG_REPLACE = ARG_ANCHOR + "    ap.add_argument('--mesh-ns',type=str,default=None,help='optional mesh divisions, e.g. 128,256; overrides --base-n')\n"
NS_ANCHOR = "    ns=tuple(args.base_n*2**j for j in range(3))\n"
NS_REPLACE = ("    if args.mesh_ns is None:\n"
"        ns=tuple(args.base_n*2**j for j in range(3))\n"
"    else:\n"
"        try:\n"
"            ns=tuple(int(t.strip()) for t in args.mesh_ns.split(','))\n"
"        except ValueError:\n"
"            ap.error('--mesh-ns must be comma-separated integers, e.g. 128,256')\n"
"        if len(ns)<2 or any(a<2 for a in ns):\n"
"            ap.error('--mesh-ns needs at least two mesh sizes, all >= 2')\n"
"        if any(b!=2*a for a,b in zip(ns[:-1],ns[1:])):\n"
"            ap.error('--mesh-ns requires successive uniform refinement by factor 2')\n")


def patch(path:Path):
    if not path.is_file():
        raise SystemExit(f'File not found: {path}')
    original=path.read_text(encoding='utf-8')
    if '--mesh-ns' in original:
        raise SystemExit('Mesh-ns option already present. Nothing changed.')
    text=original
    for name,old,new in [('parser',ARG_ANCHOR,ARG_REPLACE),('hierarchy',NS_ANCHOR,NS_REPLACE)]:
        if text.count(old)!=1:
            raise SystemExit(f'Cannot find unique {name} anchor: refusing to edit.')
        text=text.replace(old,new,1)
    ast.parse(text,filename=str(path))
    backup=path.with_name(path.name+'.before_mesh_option.bak')
    if backup.exists():
        raise SystemExit(f'Backup already exists: {backup}. Nothing changed.')
    shutil.copy2(path,backup)
    path.write_text(text,encoding='utf-8')
    print('Optional --mesh-ns implemented. Original default [64,128,256] unchanged.')
    print('Backup:',backup)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path('run_obstacle_irmtr.py'))
    patch(ap.parse_args().source)
