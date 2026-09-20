# quad3d ppo: wobble row and training-set deficit

Status board. Every task carries one of `not done`, `in progress`, `done`.
Updated as work lands. Started 2026-09-07.

Goal: six quad3d PPO levels each with 800,000 trainable flights, three of them
carrying an ungated wobble the family has never had, and eval sets for the three
new ones.

## What was decided, and why

The family has three levels, all curtain-only with `ambient: 0.0`. A ladder of
2,000 fresh box starts flown 20 times each measured what an ungated wobble buys.
Fuzzy is the share of starts whose 20 flights disagreed; the wobble-0 rows are
the published ladder, not re-flown, because `q3rl_collect.build`'s unmodified
path was checked bit-identical label for label first.

| curtain | wobble | p=1 | 0<p<1 | p=0 | mean p |
| --- | --- | --- | --- | --- | --- |
| 0.12 | 0 | 169, 8.45% | 292, 14.60% | 1539, 76.95% | 0.156 |
| 0.12 | 0.03 | 120, 6.00% | 390, 19.50% | 1490, 74.50% | 0.150 |
| 0.20 | 0 | 104, 5.20% | 302, 15.10% | 1594, 79.70% | 0.101 |
| 0.20 | 0.04 | 28, 1.40% | 406, 20.30% | 1566, 78.30% | 0.092 |
| 0.40 | 0 | 72, 3.60% | 186, 9.30% | 1742, 87.10% | 0.059 |
| 0.40 | 0.04 | 20, 1.00% | 249, 12.45% | 1731, 86.55% | 0.053 |

Chosen [user, 2026-09-07]: 0.12 at 0.03, 0.20 at 0.04, 0.40 at 0.04. The first
two hold fuzziness near 20% while mean p drops 0.150 to 0.092, so the ladder
varies difficulty at roughly constant uncertainty. That is a more useful ladder
than one where both drift together.

Recorded against 0.40 at 0.04, chosen anyway: it is beaten on both p=1 and fuzzy
by the published f_0.12 (169 and 292) and f_0.20 (104 and 302), and 12.45% is the
most the 0.40 curtain reaches at any wobble.

The LQR family's `ambient: 0.008` is decorative. At that value the ladder reads
297 / 302 / 186 fuzzy against the published 292 / 302 / 186, two of them exact.
Do not cite it as precedent for the ambient term mattering.

## Index plan

All six levels train on the same 800,000 flights:

- the 100,000 already named in `train_test_splits/shuffled_indices.txt`
- 700,000 fresh at indices 800,000 to 1,499,999

Existing levels fly only the 700,000 deficit. Their published eval sets stay
untouched: every pick's `tid` traces to a flight below index 800,000, verified
across all three levels.

The wobbled levels' eval pool is indices 0 to 799,999 minus the 100,000 training
split, which is the same 700,000 the existing levels use for theirs. They never
train on it, so no new index range is needed and nothing leaks.

## Generation speed

Measured 2026-09-07 at curtain 0.12 wobble 0.03: 109 ms per flight, 1.66 ms per
control step, `env.reset` 18.1 ms (17% of a flight). Building the env costs
4.36 s and every pool worker pays it once.

At 400 shards of 16 workers that is 6,400 builds per level, about 7.8 CPU-hours
of setup against 24 CPU-hours of flying, so 32% overhead. `compute.md` says the
CPU cap is the real concurrency limit, so that overhead is what to cut.

Fix: 100 shards instead of 400. Each worker then flies about 500 flights against
one build, dropping overhead to roughly 8%. Same starts, same seeds, same
flights, about a quarter less CPU. Wall time per task rises from ~290 s to
~940 s, still well inside the 1 hour walltime.

Rejected: making `env.reset` cheaper. It is 17% of a flight and the change would
sit inside the physics path, so it cannot be made without proving bit-identical
trajectories first. Not worth it for this collection.

## Tasks

### Patches

- [x] `done` `q3rl_collect.py`: add `--model`, `--ambient`, `--index_lo`;
      thread into `build`; record all three in the shard payload.
      (`build` itself is already patched and verified bit-identical.)
- [x] `done` `q3rl_train_reduce.py`: `wind_block()` at line 129 calls
      `resolve_noise_model('sine', f_max, None)` and writes `'ambient': 0.0` as
      a literal. Fix so the description states the wobble actually flown.
      Without this every wobbled dataset ships a description that lies, the same
      defect the wiki records against the cartpole `gaussian_signal` sets.
- [x] `done` `q3rl_train_reduce.py`: drop the `C.N_TRAIN` assert at line 59
      and replace the hardcoded 800,000 in the description text.
- [x] `done` `q3rl_collect_check.py`: take the expected flight count as an
      argument instead of `N_TRAIN = 800_000` at line 21.
- [x] `done` `q3rl_collect_train.sbatch`: pass the new flags, parameterise
      the array size, drop to 100 shards.
- [x] `done` `q3rl_fps_refly.py:44`: add model and ambient. Needed for the
      wobbled eval sets, not for training.
- [x] `done` Run `pre-commit` on every file touched.
- [x] `done` rsync to Amarel `~/scg-repo`, verify md5 both sides.

### Smoke test

- [x] `done` 20 starts at each of the six settings, confirm the wobble
      reaches the sim as `WhiteNoise` on mask [0, 1, 0].
- [x] `done` Confirm index 800,000 draws a start different from index 0.
- [x] `done` Confirm the zero-wobble path still gives identical labels.

### Training flights

- [x] `done` f_0.12 wobble 0: 700,000 flights, indices 800,000-1,499,999.
- [x] `done` f_0.20 wobble 0: 700,000 flights, same range.
- [x] `done` f_0.40 wobble 0: 700,000 flights, same range.
      Wave 1 submitted 2026-09-07: jobs 61282163/64/65, 300 tasks.
      Canary shard checked first: span [800000:807000], 1105/7000
      succeed (15.79%) against the published set's 15.59%.
      Directories are L0.12 / L0.2 / L0.4, one decimal on the last
      two, so deficit shards land beside the original 400 as
      s0400-s0499 and one reduce covers the whole 1.5M span.
- [x] `done` f_0.12 wobble 0.03: 800,000 flights.
- [x] `done` f_0.20 wobble 0.04: 800,000 flights.
- [x] `done` f_0.40 wobble 0.04: 800,000 flights.
- [x] `done` `q3rl_collect_check.py` on all six; resubmit missing shards.
      Every previous level lost 14 to 19 on the first pass.
- [x] `done` Merge each level with `q3rl_train_reduce.py`.
- [x] `done` Check collected mean p against the ladder. 0.12 at 0.03 should
      read near 0.150; previous collections matched within half a point. A miss
      means the wobble never reached the collector.

### Eval sets, wobbled levels only

- [x] `done` K decided [user, 2026-09-07]: fly the new levels at K=20 to match
      the published three, then top the whole row to 50 later. `rollout_seed` is
      pure in (pick, trial), so a later `--trial_lo 20 --trials 50` window draws
      exactly what an uninterrupted K=50 run would have drawn. No new picks, no
      GPU, and the row never sits at mixed K.
- [x] `done` Pool flights: no extra work. The wobbled levels' own 800,000
      flights over indices 0-799,999 already carry the pool; `write_splits`
      carves them into 100k train and 700k pool. I had this wrong in the plan.
      What was actually missing was the training deficit, below.
- [x] `done` Training deficit for the three wobbled levels: 700,000 each at
      indices 800,000-1,499,999, shards s0100-s0199. 200/200 complete per level.
- [x] `done` Farthest-point sample 1,000,000 picks per level from its
      own pool. Jobs 61287790/91/92, on **Amarel**, not iLab. Measured ~100 min
      per level on an A100, not the 2.6-3.6 h the older card took.
- [x] `done` `q3rl_fps_check.py` on each, before anything downstream.
- [x] `done` Re-fly every pick under that level's wind. 160 shards of 14
      CPUs; three levels at 480 tasks fits the 500-task submit cap in one pass.
- [x] `done` `q3rl_eval_reduce.py --nshards 160` on each. Fuzziness up 3.6,
      4.1 and 2.6 points against the published no-wobble levels.
- [x] `done` Save `fps_check.txt` into each level, the way the published ones
      carry it. All three read EXACT FPS; run 2026-09-08.

### Slice eval sets, wobbled levels only

Not in the original plan. Added because the wobbled levels hold 37 files where
the published levels hold 86, and the whole gap is `fps_check.txt` plus the
twelve slice lattices. A wobble row published without them looks thinner than
its neighbours for no stated reason.

- [x] `done` `q3rl_slice.py:93` called `C.build(f_max)` with no wind model, so
      every lattice cell would have been flown in curtain-only air. Fourth file
      in this pipeline with that same defect, after the collector, the train
      reducer and the eval reducer.
- [x] `done` `q3rl_slice_wobble.sbatch`, 12 planes x 4 shards. The plane list
      is read from the published `slice_description.json` files, NOT from
      `q3rl_slice50.sbatch`, whose list is stale: it still carries
      `x_dot:y_dot` at the goal, which the family page records as tried and
      dropped because still air recovers from every combination there. The
      published set has `tilt:p` in its place.
- [x] `done` Canary, job 61296404, plane x:z shard 0 of L0.12_a0.03.
      COMPLETED in 47:49, one 6.76 MB shard, cells [0:2450] at K=50, mean p
      0.819. Log confirms `ambient 0.03` reached the job.
- [ ] `not done` Full array, 48 tasks per level, 144 total. Awaiting a
      decision: it is compute nobody asked for. Real cost is 47:49 a shard,
      about 1,840 CPU-hours for the set, one wave of ~50 min wall clock
      (`main` has 230 idle nodes and 13,580 free cores against 2,304 needed).

**Read the canary against the right rows.** The lattice varies its first axis
fastest, so shard 0 covers rows 0-2449, the lowest z band, not a random sample.
The published no-wobble `x_z` lattice reads mean p 0.8146 over all 9,801 cells
but **0.8236 over those same 2,450 rows**. Against the right yardstick the
wobble lowers mean p by 0.005. Against the whole-lattice number it would have
looked like the wobble made flying easier. Same trap as the FPS shard 0 misread,
second time in this campaign, and both times the biased subset looked plausible.

**The earlier 30-40 minute estimate was wrong.** It came from job 61226833 at
~12 min a shard, which was the K=20 first pass, later topped up to 50. A K=50
shard built from scratch costs 47:49.

### Finish

- [x] `done` Write descriptions with the real wobble recorded. Verified
      2026-09-08 across all nine files: every one records `ambient` 0.03 / 0.04
      / 0.04 and `model: sine+ambient`, and the prose names the ungated push
      rather than only the number.
- [x] `done` Publish to the shared root under names that satisfy the
      glossary's naming rule (fully explicit or no parameters, never partial):
      `f_0.12_a0.03`, `f_0.20_a0.04`, `f_0.40_a0.04`. Into `ppo_800k/`, not
      `ppo/`: that directory's defining property is 800,000 trainable flights,
      which these have, and `ppo/` levels carry 100,000. 18.7 GB. Staged under
      `.staging_<name>` and renamed only after 13-entry, 26-split-file and
      `train.npz` md5 checks pass, so an interrupted transfer never looks like a
      published level.

      **Stated gap:** the three wobbled levels ship with no `slices/`. The four
      no-wobble `ppo_800k` levels inherited theirs from `ppo/`; the wobbled ones
      cannot, because slice cells must be flown under the level's own wind.
      Slices are additive later, so this does not block the publish.

      Landed 2026-09-08: 6.8 / 6.2 / 4.7 GB, `train.npz` md5 matched against
      Amarel on all three, no staging leftovers. One thing needed fixing after
      the transfer. The directories came out `drwxrwx---` against the other four
      levels' `drwxrws---`, because Linux clears setgid when a non-root user runs
      `chgrp`, and the `chgrp -R` ran after the `chmod 2770`. Without setgid,
      files added later stop inheriting `login-bekris` and st1122 loses access to
      them. `chmod g+s` restored it; all seven levels match now.
- [x] `done` Ingest into `corridor-noise.md` and the quad3d family page,
      then run `.claude/wiki_lint.py`. Reads `wiki ok: 9 pages, 8 facts
      verified against source`. Also repaired `log.md`, where an append had
      landed inside the header paragraph and hidden the 2026-09-04 fuzzy-slices
      entry from `grep '^## \['` since the day it was written.
- [x] `done` Commit the patches. Two commits on `q3-rl-corridor`, pre-commit
      clean on both. `2971e158` adds the quad3d ppo toolchain, which had never
      been tracked even though the quad2d and cartpole ones are. `64879357` is
      the wiki ingest plus the `log.md` repair.

      Scoped to the **import closure** of the collect / reduce / eval / slice
      path, 10 python files plus 5 sbatch, `submit_fps.sh` and the spec, not all
      79 untracked files. The rest are exploratory one-offs carrying 27 flake8
      errors in code this campaign never touched, and whether they belong in the
      repo is a call for a human. 61 files stay untracked.

      Verified the closure is self-contained before committing:
      `generate_quadrotor_3d_noisy.py` and `q3_corridor_common.py` are already
      tracked, and `q3rl_probe` / `q3rl_orient` are inside the closure, so
      nothing imports a file that is not there. `logs/`, `slurm_logs/` and
      `.q3_starts_cache.npy` went to `.gitignore` as outputs.

## Incident: Amarel home quota, 2026-09-07

Wave 1 lost 201 of its 300 deficit shards to a full quota on `/cache/home`.
Nothing was corrupted: `q3rl_collect` writes `<out>.tmp<pid>.npz` and then
`os.replace`, so a failed write leaves a tmp file and never a half-shard. The
completeness check read `0 short or unreadable, 201 missing`.

Two things worth carrying forward.

A file count is not a shard count. The leftover files are named
`train_L0.12_s0400.npz.tmp12345.npz`, which ends in `.npz`, so `ls *.npz | wc -l`
counted 500 per level when only 452 were real. Count
`train_L<level>_s[0-9][0-9][0-9][0-9].npz` or run the checker; never the glob.

Freed 14 GB by deleting `~/scg-repo/q3rldataset` [user, 2026-09-07]. Verified
first: 144 of its 147 files were byte-size identical to the shared root, none
were missing, and the three that differed were `dataset_description.json` files
where the shared root carries the newer key name (`state_order` replacing
`trajectory_state_order`). The Amarel copy was strictly older. `q3rlout` was
kept: `train_sk` is live for this collection, and `refly` holds the k0-20 window
shards a later K=50 top-up must sum against.

Resubmitted the 201 missing shards as jobs 61283253/54/55. All landed:
100/100 per level, 0 short, 0 tmp files.

Resolved [user, 2026-09-07] by moving collection output off home entirely.
`q3rl_collect_train.sbatch` now takes `OUTROOT`, defaulting to
`/scratch/$USER/q3rlout/train_sk`, which has 56 TB free. Wave 2 writes there.

Not cleared, and worth knowing before anyone tries: `q3dataset` (16 GB),
`q2dataset` (3.2 GB) and `cpdataset` (1.4 GB) on Amarel are NOT copies. Their
families, quad3d `f_0.000`-`f_0.072`, quad2d `f_0.000`-`f_0.200` and cartpole
`sigma_0`-`sigma_18`, appear nowhere under genMoPlan, not in the live tree and
not in `archived_data_trajectories`. Amarel holds the only copy of about 20 GB
of older collections, including the zero-mean quad2d family the corridor spec
cites at `fraction_interior` 0.122. Publish before deleting.

Wave 2 submitted: jobs 61284026/27/28, 300 tasks. The wobbled canary confirms
the wobble reaches the collector: at curtain 0.12 wobble 0.03, 1167 of 8000
starts succeed (14.59%) against the ladder's predicted mean p 0.150 and the
15.59% the same curtain gives at wobble 0. Within one standard error. It also
produced 5 timeouts where the zero-wobble canary produced none.

All six levels flew clean: 100/100 shards each, 0 short, 0 tmp leftovers,
4.5M flights. Collected success rates against the ladder:

| level | flights | rate | ladder | diff |
| --- | --- | --- | --- | --- |
| L0.12 | 1,500,000 | 15.57% | 15.6% | -0.03 |
| L0.2 | 1,500,000 | 10.52% | 10.1% | +0.42 |
| L0.4 | 1,500,000 | 6.20% | 5.9% | +0.30 |
| L0.12_a0.03 | 800,000 | 14.95% | 15.0% | -0.05 |
| L0.2_a0.04 | 800,000 | 9.50% | 9.2% | +0.30 |
| L0.4_a0.04 | 800,000 | 5.64% | 5.3% | +0.34 |

All inside the ladder's own precision at n=2000. The wobble cost 0.62 points at
curtain 0.12 against 0.6 predicted, which is the gate that would have fired if
the wobble never reached the collector.

Two more reducer patches surfaced during the merge. `--tag`, because a wobbled
level's shards are `train_L<f>_a<amb>_s####.npz` and the tag cannot be derived
from `f_max` alone. And `--n_total`, because the reducer's own completeness call
still defaulted to 800,000 and read every shard of a 1.5M merge as short. Both
are in `q3rl_train_reduce.sbatch` now.

Merge canary on L0.12_a0.03: 56.4M states, 3.0 GB, starts reproduce to 2.4e-07,
and the description carries `ambient: 0.03` with the ungated push spelled out in
`how_to_read`. That is the wind_block fix landing.

All six merged to `/scratch/dm1487/q3rldataset`: 4.0 / 3.1 / 2.3 GB for the
deficit levels at 1.5M flights, 2.9 / 2.5 / 1.7 GB for the wobbled at 800k.

Two bugs surfaced during the merge, both mine, and the second is the one worth
remembering. `check()` computed expected shard sizes as a uniform `linspace`
over the span. A deficit directory holds two tilings at once, 400 shards of
2,000 flights over `[0, 800000)` and 100 of 7,000 over `[800000, 1500000)`, and
no single linspace matches both, so all 500 read as short. It now checks each
shard against the span the shard itself records and asserts contiguity
separately.

That new check then reported a gap: one span ending at 800,000, the next
starting at 1,600,000. `q3rl_collect` already folds `index_lo` into `lo`/`hi`
before writing them, and both the new check and `merge()` added it a second
time. The wobbled levels merged fine only because their `index_lo` is 0, so
doubling zero is harmless. The gate caught a bug in the gate's own author,
which is the argument for having it.

## write_splits was discarding the whole deficit

Caught 2026-09-07 when dm1487 asked why nothing was running.

`write_splits` did `rng.permutation(len(lab))` and took the first
`TRAIN_SIZE = 100_000` as the training split. So `L0.12` merged 1.5M flights and
still marked 100,000 trainable: every deficit flight landed in the eval pool.
The board said done and the artifact was unchanged.

The second half is worse. A fresh permutation over the enlarged set scatters
flights from the published eval pool into the training split, which is exactly
the leakage the fresh-index scheme exists to prevent. The split writer would
have reintroduced it silently, and no label could contradict it.

`inherited_split()` now keeps the published eval pool verbatim and puts every
fresh index into the training split. Verified on all six: train 800,000, pool
700,000 byte-identical to published, zero overlap.

The same bug came back once more within the hour. `q3rl_train_reduce.sbatch` had
no `INHERIT` passthrough, so the first wobbled re-merge would have re-rolled the
splits again. Cancelled before those jobs wrote anything. A fix in the library
is not a fix until the thing that calls it passes the argument.

## Final shape, all six levels

| level | train.npz | trainable | eval pool | ambient |
| --- | --- | --- | --- | --- |
| L0.12 | 4.0 GB | 800,000 | 700,000 | 0 |
| L0.2 | 3.1 GB | 800,000 | 700,000 | 0 |
| L0.4 | 2.2 GB | 800,000 | 700,000 | 0 |
| L0.12_a0.03 | 5.3 GB | 800,000 | 700,000 | 0.03 |
| L0.2_a0.04 | 4.6 GB | 800,000 | 700,000 | 0.04 |
| L0.4_a0.04 | 3.1 GB | 800,000 | 700,000 | 0.04 |

Every level's pool is byte-identical to the published one and shares the same
index set, so the six are matched flight for flight. Eight times the trainable
data the family had this morning.

## The GPU step runs on Amarel, and the CPU-only torch was not a blocker

iLab was unusable on 2026-09-07: the SLURM controller was unreachable from
ilab2, ilab3, ilab4 and rlab2 alike (`Unable to contact slurm controller`), and
no login node exposes a GPU. Kerberos was fine and all four hosts took ssh, so
this was the scheduler being down rather than the ilab1 stall the skill warns
about.

Amarel's `~/envs/scg` carries torch 2.13.0+cpu, which the wiki records as the
reason GPU work goes elsewhere. That is a 103-second fix, not a blocker:

    ~/envs/scg/bin/pip install --target=/scratch/dm1487/pylibs/cu128 \
        --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0

6.6 GB on scratch, home untouched, the conda env unchanged, and jobs pick it up
with `PYTHONPATH=/scratch/dm1487/pylibs/cu128`. 2.8.0 was chosen to match the CS
side, which is the version this sampler has actually been verified against;
there is no cu128 build of 2.13.0.

**Exclude gpu017 and gpu018.** The first GPU test reported `available: False`
and looked like a broken install. It was a broken node: `nvidia-smi` on gpu017
says `Unable to determine the device handle for GPU0: Unknown Error` for one of
its four RTX 3090s while the other three answer normally, and SLURM handed the
job that card. `sinfo` flags gpu018 as `inval` but not gpu017. Excluding both,
it worked first try on an A100.

Probe before committing three levels: 20,000 picks in 2.0 min, about 167 picks a
second, self-test 500/500 identical between numpy and torch, pool 48,649,239
non-terminal states from the 700,000 held-out flights.

### Bad cards, and two fixes that were wrong first

This step keeps finding faulty GPUs. gpu017 has one of four RTX 3090s that
answers `Unable to determine the device handle for GPU0` while its siblings are
fine, and `sinfo` does not flag it. gpu029 threw `uncorrectable ECC error` 42 s
in. `sinfo` flags gpu018 `inval`. The wiki records ilab3 throwing the same ECC
error during the earlier quad3d FPS run. Two clusters, three distinct faults.

A hand-written `--exclude` is the wrong granularity: the fault is per **card**
and SLURM hands out one of two or four. gpu029's L40S cards work fine, so
banning the node throws away good hardware.

Requeue alone is not enough either, which I learned the hard way. A job bounced
six times and landed on gpu017 every time. **A broken card is an attractor**:
its three healthy siblings stay busy with other people's work, so the dead one
is permanently the only free card SLURM can offer. `scontrol update
ExcNodeList` also did not survive the requeue.

What works is a preflight (allocate and multiply a 2048x2048 tensor, about a
second) plus a **generated** exclusion file. `bounce()` appends the node to
`/scratch/dm1487/bad_gpu_nodes.txt` under flock and requeues, capped at six;
`submit_fps.sh` reads the file into `--exclude` at submit time. Nobody edits it
by hand, and emptying it re-tests everything after a repair. Requeue is nearly
free because the sampler checkpoints every 50,000 picks and a resume refuses a
checkpoint whose scaled pool differs.

### Do not raise `--gres` without checking `squeue --start`

Asking for 4 GPUs looked like the obvious speedup, since the sampler shards
across every device it sees. It cost 18 hours: SLURM put the three jobs'
estimated starts at 01:55, 12:55 and 13:55 the **next day**, because nearly
every gpu node is `mix` and four free cards on one node is rare. At `gpu:1` they
schedule in seconds. The pick loop is a single Python thread anyway, so extra
devices buy less than their count suggests; the wiki's 7-19 against 200-370
picks/s spread is CPU contention, not device count. Queue wait dominates.

## The three no-wobble levels are published

`stochastic/quadrotor3D/corridor_sine_ambient/ppo_800k/{f_0.12,f_0.20,f_0.40}`,
beside the existing `ppo/` rather than over it, so anyone on the 100k-trainable
sets keeps working. Each is self-contained: 30 new training files plus the 56
eval files copied unchanged from the published level. Inheriting the eval side
is sound because every published pick traces to a flight below index 800,000 and
the deficit sits above it, checked on all three before anything was flown.

`train.npz` md5-verified against Amarel on all three. Group `login-bekris`,
directories 2770 with setgid so later writes inherit, files 660, matching the
`ppo/` convention. st1122's group membership confirmed rather than assumed.
Sumanth DM'd with the path and the single change.

Two of my own errors on the way, both silent. Piping `rsync --info=progress2`
into `tail` kills the transfer with SIGPIPE and exits 23, leaving only the eval
side. And a verification loop using `set --` inside a `for` compared empty
strings and printed MATCH three times having read nothing. A check that can pass
without touching the data is worse than no check.

Eval sets in `ppo_800k` are K=20, identical to `ppo/`. A later K=50 top-up must
be applied to both copies or they diverge.

## The picks are done, and one number to check before claiming anything

All three pick sets pass `q3rl_fps_check.py` with EXACT FPS. Fill distance
against a uniform draw of the same size: 2.460 vs 4.495, 2.428 vs 5.802, 2.194
vs 4.161. Each `eval_fps.npz` is 92,084,230 bytes, the same size as the
published levels'.

The reflies kept to Amarel. `main` had 19,838 idle cores against the 6,720 the
job needs, iLab had 854, and iLab cannot see Amarel's scratch, so splitting
would have meant copying the picks over and the shards back for 4% more
capacity. All 480 tasks ran at once.

**A result worth not overclaiming.** The refly canary on shard 0 of
`L0.12_a0.03` read mean p 0.039 with 440 of 6,250 picks interior, 7.0%. The
published `f_0.12` at wobble 0 reads mean p 0.124 and 11.50% interior. So on the
eval picks the wobble appears to *lower* fuzziness, while the ladder measured it
*raising* fuzziness on box starts, 292 of 2,000 up to 390.

Both can be true. The ladder flies fresh starts drawn from the box. The eval set
is farthest-point picks from mid-flight states, and the family page records that
most of those are already committed to failing. Wind added to a doomed flight
kills it sooner rather than making it uncertain.

**Settled, and the alarm was wrong.** The full reduce says the wobble raises
eval-set fuzziness on every level, the same direction the ladder predicted:

| level | uncertain | published, no wobble | always succeed |
| --- | --- | --- | --- |
| f_0.12_a0.03 | 151,389, 15.14% | 11.50% | 50,790 |
| f_0.20_a0.04 | 155,689, 15.57% | 11.46% | 13,180 |
| f_0.40_a0.04 | 101,771, 10.18% | 7.54% | 7,871 |

Gains of 3.6, 4.1 and 2.6 points. `agreement_with_source_label` 0.965, 0.969,
0.982.

Why the canary misled: shard 0 covers picks 0 to 6,250 of a farthest-point
**ordering**, so it holds the earliest and most spread-out picks. It is biased
by construction, not a small random sample. Never read a single FPS shard as
representative of the set.

A third file carried the ambient defect. `q3rl_eval_reduce.py` called
`TR.wind_block(a.f_max)` with one argument, so `ambient` defaulted to 0 and every
wobbled eval description would have claimed no wobble. Patched to take
`--model`/`--ambient` and to name the ungated push in its prose.

## Costs

- 6.6M flights total. At 109 ms each and 100 shards of 16 workers, roughly an
  hour of allocation across the array jobs. No GPU.
- 8 to 11 GPU-hours for the three farthest-point runs, the entire GPU bill.
- About 10 GB more on disk against the 21 GB the stochastic tree holds now.

## Open question, not blocking

`train.npz` is the right format. Every stochastic family ships it, and st1122's
2026-09-06 results cover all four quad3d PPO levels, so a reader exists. The
genMoPlan checkout at `local_dynamics/genMoPlan` is `main` from 2026-03-19 and
reads only a `trajectories/` directory, but that is a stale local copy, not the
pipeline. Worth confirming which branch and loader those results came from so
the new levels ship in the layout that reader expects.
