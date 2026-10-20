"""
Value-matched import of an earlier seeded-rate run of the SAME model on a
DIFFERENT grid, for scripts/submit_model14_15_rate.sh: model14's v2 grid
(gamma_1 in [0, 2] step 0.1 x h in [0, 4.8] step 0.4, 21 x 13) into its v3
grid (gamma_1 in [0, 3] step 0.2 x h in [0, 1.3] step 0.1, 16 x 14), where the
44 shared points (gamma_1 = 0, 0.2, .., 2; h = 0, 0.4, 0.8, 1.2) sit at other
indices.

scripts/rate_seeded_import.py needs the source grid to be the leading block of
the target grid (same index -> same point) and copies every round file.
Neither works here: the indices differ, and copied round files would carry
the source's round numbers into the target (the v2 run reached rrefine round
99, the cap, so the target could never run a round of its own).  Instead, per
source point whose stored (p1, p2) lies on the target grid:

  <model>_seeded/, <model>_seeded_d12/   ONE base file pt_<ix>_<iy>.npz: the
      source base file with rate_<m> / S_<m> / label_<m> replaced by the
      source point's best known frame over all its files (worker, xeval,
      qrefine, margin, rrefine: model10_seeded_qrefine_worker.best_known),
      m = 4, 8 (resp. 12).  No round files, so the target's rounds start at 1;
      its seed stages skip the imported points and its xeval / refine stages
      spread the imported frames to the new neighbours.
  <model>_product/, <model>_8q/          the base file, re-indexed.

Points are matched by the parameter VALUES stored in the files (keys
<p1_name> / <p2_name>, else p1 / p2), never by index.  The bond dimension
('dim') must match, and every stored spectral abscissa ('floor', seeded and
product files) is recomputed from the target model at the matched point: a
mismatch (a source run of another model definition) aborts the import before
anything is written.  Files are written atomically and never overwrite a
target file; the source run is only read.

Usage:
    python scripts/rate_seeded_remap_import.py --model model14 \\
        --src_dir results_model14_v2_seeded [--dry_run]
"""
from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS                               # noqa: E402
from framability_rate import spectral_abscissa                           # noqa: E402
from model10_seeded_worker import pt_dir_name, generator                  # noqa: E402
from model10_d12_worker import d12_dir                                    # noqa: E402
from model10_seeded_qrefine_worker import best_known                      # noqa: E402
from model4_product_rate_worker import prod_tag                           # noqa: E402
from model4_manybody_worker import mb_tag                                 # noqa: E402

BASE_RE = re.compile(r'pt_(\d{3})_(\d{3})\.npz')
VAL_TOL = 1e-9             # grid values are rounded to 10 digits (_arange)
FLOOR_TOL = 1e-8


def stored_point(d, spec) -> tuple[float, float] | None:
    for a, b in ((spec.p1_name, spec.p2_name), ('p1', 'p2')):
        if a in d.files and b in d.files:
            return float(d[a]), float(d[b])
    return None


def index_of(vals: np.ndarray, v: float) -> int | None:
    i = int(np.argmin(np.abs(vals - v)))
    return i if abs(vals[i] - v) <= VAL_TOL else None


def matches(src_pt: Path, model: str, stride: int, dim: int) -> list:
    """[(source base file, ixo, iyo, ix, iy, p1, p2)] of the source points on
    the target grid at `stride`; exits on a bond-dimension or generator
    mismatch."""
    spec = MODELS[model]
    p1v = np.asarray(spec.p1_vals[::stride], float)
    p2v = np.asarray(spec.p2_vals[::stride], float)
    out = []
    for f in sorted(src_pt.iterdir()):
        mt = BASE_RE.fullmatch(f.name)
        if not mt:
            continue
        try:
            d = np.load(f, allow_pickle=True)
        except Exception as e:                              # noqa: BLE001
            print(f'  warning: {f}: {e}; skipped', flush=True)
            continue
        pp = stored_point(d, spec)
        if pp is None:
            sys.exit(f'ERROR: {f} stores no parameter values; not importing')
        ix, iy = index_of(p1v, pp[0]), index_of(p2v, pp[1])
        if ix is None or iy is None:
            continue
        if 'dim' in d.files and int(d['dim']) != dim:
            sys.exit(f'ERROR: {f} was computed at dim={int(d["dim"])}, the '
                     f'target uses dim={dim}; not importing')
        if 'floor' in d.files:
            fl = spectral_abscissa(generator(pp[0], pp[1], dim, model))
            if abs(fl - float(d['floor'])) > FLOOR_TOL:
                sys.exit(f'ERROR: {f}: stored spectral abscissa {float(d["floor"]):.10g} '
                         f'!= {fl:.10g} of {model} at {pp}; the source is another '
                         'model definition, not importing')
        out.append((f, int(mt.group(1)), int(mt.group(2)), ix, iy, pp[0], pp[1]))
    return out


def write_atomic(path: Path, content: dict) -> None:
    tmp = path.with_name(path.name[:-len('.npz')] + '.tmp.npz')
    np.savez(tmp, **content)
    os.replace(tmp, path)                     # never leave a half-written file


def import_folder(src_pt: Path, dst_pt: Path, model: str, stride: int, dim: int,
                  d_exts, dry_run: bool, tag: str) -> int:
    """Import one folder; d_exts: rate keys to consolidate (None: copy)."""
    if not src_pt.is_dir():
        print(f'{tag} {src_pt}: not found, skipped', flush=True)
        return 0
    pts = matches(src_pt, model, stride, dim)
    n_new = n_old = 0
    for f, ixo, iyo, ix, iy, p1, p2 in pts:
        g = dst_pt / f'pt_{ix:03d}_{iy:03d}.npz'
        if g.exists():
            n_old += 1
            continue
        n_new += 1
        if dry_run:
            continue
        d = np.load(f, allow_pickle=True)
        content = {k: d[k] for k in d.files}
        txt = []
        for m in d_exts or ():
            v, S, lab = best_known(src_pt, ixo, iyo, f'rate_{m}', f'S_{m}')
            if S is None:
                continue
            if f'rate_{m}' in content:
                content[f'import_rate_{m}_base'] = float(content[f'rate_{m}'])
            content[f'rate_{m}'] = v
            content[f'S_{m}'] = S
            content[f'label_{m}'] = f'import {lab}'.strip()
            txt.append(f'd{m}={v:.5f}')
        content.update(model=model, ix=ix, iy=iy, stride=stride,
                       import_src=str(f), import_src_ix=ixo, import_src_iy=iyo)
        dst_pt.mkdir(parents=True, exist_ok=True)
        write_atomic(g, content)
        if txt:
            print(f'{tag}   ({p1:g}, {p2:g}): {f.name} -> {g.name}  {"  ".join(txt)}',
                  flush=True)
    print(f'{tag} {src_pt} -> {dst_pt}: {len(pts)} source point(s) on the target '
          f'grid; {n_new} {"to import" if dry_run else "imported"}, {n_old} already '
          'there', flush=True)
    return n_new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True, choices=sorted(MODELS),
                    help='model of both runs (its CURRENT grid is the target)')
    ap.add_argument('--src_dir', type=str, required=True,
                    help='root of the source run (e.g. results_model14_v2_seeded)')
    ap.add_argument('--src_name', type=str, default=None,
                    help='model name in the source subfolders (default --model)')
    ap.add_argument('--out_dir', type=str, default=None,
                    help='target root (default results_<model>_seeded)')
    ap.add_argument('--stride', type=int, default=1,
                    help='target stride of the seeded / d_ext = 12 folders')
    ap.add_argument('--prod_stride', type=int, default=None,
                    help='target stride of the product folder (default --stride)')
    ap.add_argument('--mb_stride', type=int, default=1,
                    help='target stride of the 8-qubit folder')
    ap.add_argument('--dim', type=int, default=None,
                    help="bond dimension of both runs (default: the model's)")
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()
    model = args.model
    src_name = args.src_name or model
    dim = args.dim if args.dim is not None else MODELS[model].dim
    src_root = Path(args.src_dir)
    dst_root = Path(args.out_dir or f'results_{model}_seeded')
    if not src_root.is_dir():
        sys.exit(f'ERROR: source run {src_root} not found')
    if src_root.resolve() == dst_root.resolve():
        sys.exit('ERROR: source and target roots are the same folder')
    prod_stride = args.prod_stride or args.stride
    tag = f'[remap-import {src_root} -> {dst_root}]'

    # (source, target, target stride, rate keys to consolidate); the source
    # seeded folders are the unstrided ones
    folders = [(src_root / pt_dir_name(1, src_name),
                dst_root / pt_dir_name(args.stride, model), args.stride, (4, 8)),
               (Path(d12_dir(str(src_root), 1, src_name)),
                Path(d12_dir(str(dst_root), args.stride, model)), args.stride, (12,)),
               (src_root / prod_tag(src_name), dst_root / prod_tag(model),
                prod_stride, None),
               (src_root / mb_tag(src_name), dst_root / mb_tag(model),
                args.mb_stride, None)]
    # every check first (matches() exits on a mismatch), then write
    for src_pt, _, stride, _ in folders:
        if src_pt.is_dir():
            matches(src_pt, model, stride, dim)
    tot = sum(import_folder(s, t, model, k, dim, ds, args.dry_run, tag)
              for s, t, k, ds in folders)
    print(f'{tag} {"would import" if args.dry_run else "imported"} {tot} '
          'file(s) in total', flush=True)


if __name__ == '__main__':
    main()
