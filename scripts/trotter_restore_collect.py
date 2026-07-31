"""
Merge the per-chunk manifest fragments written by
scripts/trotter_restore_confirm_worker.py into a single recompute manifest that
scripts/trotter_recompute_worker.py (Stage 2) consumes.

The output schema is identical to the one scripts/trotter_alt_confirm.py emits
(rows = [model_idx, ix, iy, de, p1, p2, unconf_fra, restored/conf_fra,
was_claimed], plus model_names), so Stage 2 needs no change.

    python scripts/trotter_restore_collect.py \
        --frag_glob 'restore_manifest_chunk*.npz' \
        --out_manifest recompute_manifest
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--frag_glob', type=str,
                   default='restore_manifest_chunk*.npz',
                   help='glob for the per-chunk manifest fragments')
    p.add_argument('--out_manifest', type=str, default='recompute_manifest',
                   help='basename for the merged <name>.npz / <name>.txt')
    args = p.parse_args()

    frags = sorted(glob.glob(args.frag_glob))
    if not frags:
        print(f'No fragments match {args.frag_glob!r}.')
        return

    all_rows = []
    for fr in frags:
        z = np.load(fr, allow_pickle=True)
        r = z['rows']
        if r.size:
            all_rows.append(r.reshape(-1, 9))
        z.close()

    if not all_rows:
        print(f'{len(frags)} fragments, but every stored old frame certifies '
              '-- nothing to recompute. No manifest written.')
        return

    rows = np.vstack(all_rows)
    # De-duplicate on (model_idx, ix, iy, de); keep the first occurrence.
    seen: set = set()
    keep = []
    for row in rows:
        key = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        if key not in seen:
            seen.add(key)
            keep.append(row)
    rows = np.array(keep, dtype=float)

    names = list(MODELS)
    np.savez(f'{args.out_manifest}.npz', rows=rows,
             columns=np.array(['model_idx', 'ix', 'iy', 'de', 'p1', 'p2',
                               'unconf_fra', 'conf_fra', 'was_claimed']),
             model_names=np.array(names))
    with open(f'{args.out_manifest}.txt', 'w') as fh:
        fh.write('# model ix iy d_ext p1 p2 unconf_fra restored_fra was_claimed\n')
        for row in rows:
            m = names[int(row[0])]
            cf = 'inf' if not np.isfinite(row[7]) else f'{row[7]:.6f}'
            fh.write(f'{m} {int(row[1])} {int(row[2])} {int(row[3])} '
                     f'{row[4]:.4f} {row[5]:.4f} {row[6]:.6f} {cf} '
                     f'{int(row[8])}\n')

    # Per-model / per-point summary.
    from collections import Counter
    by_model = Counter(names[int(r[0])] for r in rows)
    pts = {(int(r[0]), int(r[1]), int(r[2])) for r in rows}
    print(f'Merged {len(frags)} fragments -> {len(rows)} (point,d_ext) records, '
          f'{len(pts)} unique points needing recompute.')
    for m in names:
        if by_model.get(m):
            print(f'  {m}: {by_model[m]} (point,d_ext)')
    print(f'Wrote {args.out_manifest}.npz / {args.out_manifest}.txt')


if __name__ == '__main__':
    main()
