# quad3d twin-curtain collection, on Amarel

Status: proposed, 2026-08-19. Nothing here has been run.

Follows the quad2d corridor campaign. Same disturbance family, same shard
schema, same reducer, so the tooling carries over. What does not carry over is
the slice design, for a reason measured below.

## 1. Force levels

The 8-level sweep (jobs 212096-212103, 400 starts, 30 flights each, ambient
0.008) measured against a deterministic baseline of 0.2197 success:

| F_max (N) | success | retention | fuzzy | rescue side | hazard side |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 0.213 | 0.97 | 2.8% | 0.196 | 0.232 |
| 0.05 | 0.210 | 0.95 | 4.8% | 0.181 | 0.242 |
| 0.08 | 0.208 | 0.95 | 6.0% | 0.183 | 0.237 |
| 0.13 | 0.196 | 0.89 | 8.3% | 0.178 | 0.217 |
| 0.20 | 0.178 | 0.81 | 12.8% | 0.170 | 0.188 |
| 0.30 | 0.131 | 0.60 | 17.3% | 0.131 | 0.132 |
| 0.50 | 0.068 | 0.31 | 14.0% | 0.062 | 0.075 |

Retention is success at that level divided by success at F_max 0. quad2d's two
published configs sit at 0.74 and 0.76, so the quad2d yardstick falls between
0.20 N and 0.30 N and no measured level lands on it.

**Recommendation: measure 0.25 N first, then collect two of {0.20, 0.25, 0.30}.**
One sweep job, about 15 minutes on the numbers from this morning's eight. If
0.25 N lands near retention 0.75 it becomes the direct quad2d analogue and the
pair is 0.25 with one of its neighbours.

Two things argue against taking 0.30 N blind. Fuzziness peaks there at 17.3%
and then falls to 14.0% at 0.50 N, so 0.30 is already near the turn where the
force stops creating uncertainty and starts only destroying runs. And the
rescue and hazard sides converge there, 0.131 against 0.132, so the twin-curtain
contrast vanishes exactly where the fuzziness is highest. At 0.20 N they still
separate, 0.170 against 0.188.

Ambient stays at 0.008 throughout, the value the sweep used.

## 2. Slices: the quad2d design does not transfer

quad2d sliced over (x, z) at fixed attitude and rates, and those panels carried
real structure. They do not here.

Measured over 300,000 rows of the shipped deterministic eval set, success rate
by decile of each coordinate:

| coordinate | low decile | middle | high decile |
| --- | --- | --- | --- |
| qw | 0.08 | 0.13 | 0.67 |
| r | 0.12 | 0.49 | 0.12 |
| p, q | 0.11 | 0.42 | 0.11 |
| z_dot | 0.02 | 0.31 | 0.13 |
| x, y, z | 0.13 | 0.27 | 0.13 |

Attitude and body rates decide the outcome. Position moves it from 0.13 to
0.27 and no further. So a (x, z) panel at fixed attitude comes out close to
uniform, and a slice with no deterministic boundary in it shows nothing about
how the corridor moves that boundary.

The other blocker: the quad3d eval set is 1,000,000 randomly drawn 13-D points,
not a lattice. Every coordinate has tens of thousands of distinct values, so a
slice cannot be cut by filtering. It has to be generated, which is what
q2_corridor_slice.py already does with its own grid_states().

**Proposed slices**, all with x on the horizontal axis so the curtain crossing
is visible, and a boundary-carrying coordinate on the vertical:

- (x, qw), the steepest gradient. Sweeps from near-inverted to level.
- (x, r), the sharpest shape. Success peaks near zero yaw rate and falls off
  both sides, so the panel holds two boundaries rather than one.
- (x, z_dot), for a case where the drone is already falling.

**Gate before spending collection compute:** generate each candidate slice at
F_max 0 and check it contains a boundary, meaning `fraction_interior` is not
0 and the panel is neither all-success nor all-failure. A slice that fails this
gate is discarded, not collected. This is cheap: one slice is a 61x43 grid at
1 flight per cell.

## 3. Sizing on Amarel

The full eval grid is 1,000,000 states at 20 flights, so 20,000,000 flights per
config at a 2000-step horizon. quad2d's comparable pass was 9,375,960 flights
at 1200 steps and took an afternoon on roughly 900 cores.

Amarel's `main` partition measured 27,320 idle cores of 29,392 on 2026-08-19.
The per-user cap is 6,720 CPUs running and 500 queued jobs, QOS `main`, 3-day
walltime. Array tasks count individually against the 500.

Shape that fits: **one array of 500 tasks at 13 CPUs each**, about 6.5k CPUs,
which is the documented cap. Or 250x10 for a politer 2,500. The user's fair-use
note says deliberate bursts go wide but stay under 3 hours, and to leave
headroom against the idle pool rather than taking more than a quarter of it.

## 4. What has to happen first

1. **Copy the data.** `q3_corridor_collect.py` opens exactly one file,
   `eval_states.txt`, 249 MB. Not the 7.1 GB directory. Point `Q3_DET_DIR` at
   the copy on `/scratch/dm1487`.
2. **Pull on Amarel.** `~/scg-repo` is on the same branch and the same remote,
   which is already `dhruvmetha/safe-control-gym`, and the fork now has every
   commit. 18 modified files there match 18 modified here, so they are one set
   of edits made twice, not divergence. 2,681 untracked files are old scratch
   scripts and logs that git never tracked and a pull will not touch.
3. **Check the env.** `~/envs/scg` imports pybullet and numpy. Nothing else has
   been verified.
4. **Test `shard_train`.** Untested. It calls `sampler_starts()`, which draws
   800k starts and costs 17 s unless the cache exists.
5. **Run the 0.25 N sweep**, then pick the pair.
6. **Gate the slices** as above.
7. Only then collect.

## 5. Carried over unchanged

`q3_corridor_collect.py` writes the same shard schema as quad2d, so
`q2_corridor_topup.py`, `q2_corridor_verify.py` and the reducer's window
summing all work on quad3d shards with a path change. The trial-window
mechanism means a K=20 to K=50 top-up stays available without recollecting.

The reachability shortcut is gated on `ambient_on` from the start. The quad2d
version was not, and it silently spoiled 40.9% of that campaign's eval labels;
see the FALSIFIED block in the quad2d corridor design spec. The same trap is
available here because the curtain is gated on x, so "the disturbance cannot
reach this start" reads just as convincingly and fails for the same reason.

## 6. Open questions

- Which two force levels, pending the 0.25 N measurement.
- Whether the rescue and hazard contrast collapsing at 0.30 N disqualifies it.
- Train split size. quad2d used 500,000 trajectories; quad3d's `N_TRAIN` is a
  different number and has not been checked against this campaign's needs.
