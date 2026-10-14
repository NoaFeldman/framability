"""
Stage 0 of scripts/submit_rate_seeded.sh: import an earlier seeded-rate run
whose grid is a leading block of this run's grid, so that only the new points
are computed.  Example: model3 (gamma <= 10, 51 x 51) into model3x (gamma <= 16,
81 x 51; trotter_lindbladian_scan.MODELS).

Valid when, on the strided grid of every folder, the source model's p1 / p2
values equal the target's first len(p1) / len(p2) values (same index -> same
point), and the two models have the same bond generator on the source grid
(checked at a few points).  A source file pt_<ix>_<iy>[_<tag>].npz is then the
target's file of the same name: worker, xeval and every round file
(_qrefine_rNN, _margin_rNN, _rrefine_rNN) are copied unchanged, so the target
run's stages skip the imported points and its round numbers continue after the
imported rounds.

Folders (source -> target, under --src_dir / --out_dir):
    <src>_seeded[_s<k>]       -> <model>_seeded[_s<k>]       (--stride)
    <src>_seeded_d12[_s<k>]   -> <model>_seeded_d12[_s<k>]   (--stride)
    <src>_product             -> <model>_product              (--prod_stride)
    <src>_8q                  -> <model>_8q                   (--mb_stride)
The product / 8q folders are not stride-tagged, so they are imported only if
the stride stored in their files is the one this run uses.  Missing source
folders are skipped.  Files are COPIED (the source run is never
touched) and never overwrite a target file.  Leftover *.tmp.npz files of
killed jobs are not copied.  Import only a finished source run: a source round
file written after the import may collide with the target's own round of the
same number (the target's file is kept).

Usage:
    python scripts/rate_seeded_import.py --src_model model3 \\
        --src_dir results_model3_seeded --model model3x --out_dir results_model3x_seeded
    (--dry_run: count only)
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS, build_bond_lindbladian      # noqa: E402
from model10_seeded_worker import pt_dir_name                             # noqa: E402
from model10_d12_worker import d12_dir                                    # noqa: E402
from model4_product_rate_worker import prod_tag                           # noqa: E402
from model4_manybody_worker import mb_tag                                 # noqa: E402

PT_RE = re.compile(r'pt_\d{3}_\d{3}(_.+)?\.npz')


def check_grid(src: str, dst: str, stride: int) -> tuple[int, int]:
    """(nx, ny) of the source grid at `stride`; exits unless it is the leading
    block of the target grid at the same stride."""
    out = []
    for ax in ('p1', 'p2'):
        a = np.asarray(getattr(MODELS[src], f'{ax}_vals')[::stride], float)
        b = np.asarray(getattr(MODELS[dst], f'{ax}_vals')[::stride], float)
        if len(a) > len(b) or not np.allclose(a, b[:len(a)], rtol=0, atol=1e-9):
            sys.exit(f'ERROR: {src} {ax} grid (stride {stride}) is not the leading '
                     f'block of the {dst} grid: {a[:3]}..{a[-1]} vs {b[:3]}..{b[-1]}')
        out.append(len(a))
    return out[0], out[1]


def check_generator(src: str, dst: str, dim: int) -> None:
    """Exit unless the two models' bond generators agree at a few source-grid
    points (corners and centre)."""
    p1, p2 = MODELS[src].p1_vals, MODELS[src].p2_vals
    for i, j in [(0, 0), (len(p1) - 1, len(p2) - 1), (len(p1) // 2, len(p2) // 2),
                 (len(p1) - 1, 0), (0, len(p2) - 1)]:
        La = build_bond_lindbladian(*MODELS[src].build(p1[i], p2[j]), dim)
        Lb = build_bond_lindbladian(*MODELS[dst].build(p1[i], p2[j]), dim)
        if not np.allclose(La, Lb, rtol=0, atol=1e-12):
            sys.exit(f'ERROR: {src} and {dst} bond generators differ at '
                     f'({p1[i]}, {p2[j]}); not importing')


def file_stride(folder: Path) -> int | None:
    """Grid stride stored in the folder's first per-point file (None if the
    folder has none or the key is missing)."""
    for f in sorted(folder.glob('pt_[0-9][0-9][0-9]_[0-9][0-9][0-9].npz')):
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:                                   # noqa: BLE001
            continue
        return int(d['stride']) if 'stride' in d.files else None
    return None


def check_dim(folder: Path, dim: int) -> None:
    """Exit if the folder's first file carrying a 'dim' key was computed at a
    different bond dimension."""
    for f in sorted(folder.glob('pt_[0-9][0-9][0-9]_[0-9][0-9][0-9].npz')):
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:                                   # noqa: BLE001
            continue
        if 'dim' in d.files:
            if int(d['dim']) != dim:
                sys.exit(f'ERROR: {folder} was computed at dim={int(d["dim"])}, '
                         f'this run uses dim={dim}; not importing')
            return


def copy_folder(src_pt: Path, dst_pt: Path, nx: int, ny: int,
                dry_run: bool) -> tuple[int, int]:
    """Copy every per-point file of src_pt with ix < nx, iy < ny that dst_pt
    does not hold yet; returns (copied, already present)."""
    n_new = n_old = 0
    if not dry_run:
        dst_pt.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_pt.iterdir()):
        if not PT_RE.fullmatch(f.name) or f.name.endswith('.tmp.npz'):
            continue
        ix, iy = int(f.name[3:6]), int(f.name[7:10])
        if ix >= nx or iy >= ny:
            sys.exit(f'ERROR: {f} lies outside the {nx} x {ny} source grid')
        g = dst_pt / f.name
        if g.exists():
            n_old += 1
            continue
        if not dry_run:
            tmp = g.with_name(g.name + '.part')
            shutil.copy2(f, tmp)
            tmp.replace(g)                    # never leave a half-copied file
        n_new += 1
    return n_new, n_old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_model', type=str, required=True, choices=sorted(MODELS))
    ap.add_argument('--src_dir', type=str, required=True,
                    help='root of the source run (e.g. results_model3_seeded)')
    ap.add_argument('--model', type=str, required=True, choices=sorted(MODELS),
                    help='target model (e.g. model3x)')
    ap.add_argument('--out_dir', type=str, default=None,
                    help='target root (default results_<model>_seeded)')
    ap.add_argument('--stride', type=int, default=1,
                    help='stride of the seeded / d_ext = 12 folders')
    ap.add_argument('--prod_stride', type=int, default=None,
                    help='stride of the product-state folder (default --stride)')
    ap.add_argument('--mb_stride', type=int, default=1,
                    help='stride of the 8-qubit folder')
    ap.add_argument('--dim', type=int, default=None,
                    help="bond dimension of the target run (default: the model's)")
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()
    src, dst = args.src_model, args.model
    if src == dst:
        sys.exit('ERROR: --src_model and --model are the same model')
    dim = args.dim if args.dim is not None else MODELS[dst].dim
    src_root = Path(args.src_dir)
    dst_root = Path(args.out_dir or f'results_{dst}_seeded')
    if not src_root.is_dir():
        sys.exit(f'ERROR: source run {src_root} not found')
    check_generator(src, dst, dim)

    prod_stride = args.prod_stride or args.stride
    # (source, target, stride, stride stored in the files must match)
    folders = [(src_root / pt_dir_name(args.stride, src),
                dst_root / pt_dir_name(args.stride, dst), args.stride, False),
               (d12_dir(str(src_root), args.stride, src),
                d12_dir(str(dst_root), args.stride, dst), args.stride, False),
               (src_root / prod_tag(src), dst_root / prod_tag(dst), prod_stride, True),
               (src_root / mb_tag(src), dst_root / mb_tag(dst), args.mb_stride, True)]
    tot = 0
    for src_pt, dst_pt, stride, untagged in folders:
        tag = f'[import {src} -> {dst}] {src_pt}'
        if not src_pt.is_dir():
            print(f'{tag}: not found, skipped', flush=True)
            continue
        if untagged:
            k = file_stride(src_pt)
            if k is None:
                print(f'{tag}: no per-point files with a stride, skipped', flush=True)
                continue
            if k != stride:
                print(f'{tag}: computed at stride {k}, this run uses {stride}; '
                      'skipped', flush=True)
                continue
        nx, ny = check_grid(src, dst, stride)
        if src_pt.name != mb_tag(src):          # 8q files: no bond dimension
            check_dim(src_pt, dim)
        n_new, n_old = copy_folder(src_pt, dst_pt, nx, ny, args.dry_run)
        tot += n_new
        print(f'{tag} -> {dst_pt} (stride {stride}, {nx} x {ny} block): {n_new} file(s) '
              f'{"to copy" if args.dry_run else "copied"}, {n_old} already there',
              flush=True)
    print(f'[import {src} -> {dst}] {"would copy" if args.dry_run else "copied"} '
          f'{tot} file(s) in total', flush=True)


if __name__ == '__main__':
    main()
