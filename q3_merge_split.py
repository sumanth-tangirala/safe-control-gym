'''Merge the 16 sub-shards of quad3d f=0.072 chunk 12 into one shard file.

The chunk was rerun split 16 ways (shards 192-207 of 896) because as a single
task it took ~10h and had already been lost twice -- once to the 12h wall, once
to a halk node that wrote nothing. The boundaries divide exactly: shards 192-207
of 896 cover [214285, 232142], identical to chunk 12 of 56.
'''
import glob
import os
import sys

import numpy as np

SPLIT = os.path.expanduser('~/scg-repo/q3out/split')
OUT = os.path.expanduser('~/scg-repo/q3out/eval/L0.072_s012.npz')
EXPECT_LO, EXPECT_HI = 214285, 232142

files = sorted(glob.glob(os.path.join(SPLIT, 'L0.072_p*.npz')))
if len(files) != 16:
    print(f'have {len(files)}/16 sub-shards, not merging yet')
    sys.exit(1)

parts = [np.load(f) for f in files]
order = np.argsort([int(p['lo']) for p in parts])
parts = [parts[i] for i in order]
lo, hi = int(parts[0]['lo']), int(parts[-1]['hi'])
assert (lo, hi) == (EXPECT_LO, EXPECT_HI), f'range {lo}:{hi} != {EXPECT_LO}:{EXPECT_HI}'
# contiguity: each piece must start where the previous ended
for a, b in zip(parts, parts[1:]):
    assert int(a['hi']) == int(b['lo']), 'sub-shards are not contiguous'
trials = {int(p['trials']) for p in parts}
assert trials == {100}, f'inconsistent trials {trials}'

starts = np.concatenate([p['starts'] for p in parts])
hits = np.concatenate([p['hits'] for p in parts])
det = np.concatenate([p['det_labels'] for p in parts])
assert len(starts) == hi - lo, f'{len(starts)} rows for range of {hi - lo}'

tmp = OUT + f'.tmp{os.getpid()}.npz'
np.savez(tmp, starts=starts, hits=hits, det_labels=det, trials=100,
         lo=lo, hi=hi, level=0.072, mechanism='dynamics')
os.replace(tmp, OUT)
print(f'merged {len(files)} sub-shards -> {OUT}')
print(f'  {len(starts)} states, p_success mean {hits.sum() / (len(hits) * 100):.4f}')
