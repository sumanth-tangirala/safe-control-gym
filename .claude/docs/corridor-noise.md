# Corridor noise (quad2d, quad3d)

Load when generating, reading, or extending the `corridor_sine_ambient` families,
or when any collection needs a top-up to a higher trial count.

A different mechanism from every other stochastic family here. The others are
uniform in space and redrawn every control step. This one is **gated on one
coordinate** and **coherent in time**: a rollout draws its gust amplitude and
phase once at reset and then holds them, so the force is a deterministic function
of position and time for that flight.

Design specs: `docs/superpowers/specs/2026-08-17-quad2d-altitude-corridor-design.md`
and `plans/quad3d-corridor-collection.md`.

## The law

```
F = sigma(gate_coord) * (0.5 + 0.5*A*sin(2*pi*t/T + phi)) + N(0, ambient)
A ~ U(0, 1), phi ~ U(-pi, pi), both drawn once per rollout and held
T = 2.0 s, at ctrl_freq 100 Hz
```

The bracket never goes negative, so the corridor term is **one-sided**. Its mean
over a flight is `sigma/2`, not `sigma` — the envelope is a draw bound, not a
typical value. The ambient term is a separate, ungated, zero-mean Gaussian
redrawn every step everywhere in the arena, including inside the calm region
around the goal.

## Geometry, per system

Both gate on one coordinate and push along a **different** one, so a crossing is
shoved along the face of the layer rather than back out of it.

| | quad2d | quad3d |
| --- | --- | --- |
| gate coordinate | `z`, altitude | `x` |
| push direction | `+x`, horizontal | `+y` |
| shape | one gaussian, centre 0.55 m | **two** gaussians, centres +-0.9 m |
| width | 0.12 | 0.25 |
| band, 1% of peak | z in [0.186, 0.914], 0.728 m | x in [0.141, 1.659] and mirror, 1.517 m each |
| goal | (0, 1.0), tol 0.2 | (0, 0, 1), tol 0.05 |
| controller | `rl`, safe-explorer PPO | `lqr` |
| horizon | 1200 steps, 12 s | see the split-deadline defect below |
| eval states | 489,789 | 1,000,000 |
| train trajectories | 500,000 | 800,000 |

quad3d's two curtains draw **independently**, so their envelopes add. `sigma(x)`
returns the sum, and the mean force at `x` is `sigma(x)/2`. `resolve_noise_model`
emits the +X_C curtain first and the -X_C second, a fixed order, because
`DisturbanceList` spawns child RNG streams in list order. Reordering that list
changes which stream each curtain draws from and breaks reproducibility.

### The push axis is not the gate axis, and the mask hides it

`mask` multiplies the `disturb_force` vector elementwise and `quadrotor.py`
passes that vector to PyBullet unchanged. quad2d uses `mask [1, 0]`, and for
`TWO_D` index 0 maps to x. quad3d uses `mask [0, 1, 0]`, and for `THREE_D` index
1 is **y**, not x.

Measured 2026-08-19 by parking the drone at a curtain peak and reading the force
handed to PyBullet: `[0, 0.068, 0]`. The registered function is still called
`altitude_gated_sine`, inherited from quad2d where the gate really was altitude.
On quad3d that name is legacy; `gated_on` and `direction` in the description are
the authoritative fields.

## The collected family

```
DATA_ROOT/stochastic/quadrotor2D/corridor_sine_ambient/rl/{baseline,sharp,smooth}/
DATA_ROOT/stochastic/quadrotor3D/corridor_sine_ambient/lqr/{f_0.25,f_0.30}/
```

| config | f_max | ambient | f_max as share of weight | fuzzy @ K=100 |
| --- | --- | --- | --- | --- |
| quad2d `baseline` | 0 | none | 0% | 0.00% |
| quad2d `sharp` | 0.08 N | 0.06 N | 30.2% | 9.29% |
| quad2d `smooth` | 0.05 N | 0.09 N | 18.9% | 11.55% |
| quad3d `f_0.25` | 0.25 N | 0.008 N | 94.4% | 16.33% |
| quad3d `f_0.30` | 0.30 N | 0.008 N | 113.3% | 18.72% |

Drone weight is 0.265 N on both systems (0.027 kg). The quad3d envelope peak is
about the drone's own weight, and at `f_0.30` it exceeds it: crossing a curtain
is being pushed sideways roughly as hard as gravity pulls down.

**`smooth` is fuzzier than `sharp` despite the weaker gust**, 11.55% against
9.29%. Its ambient term is larger and ungated, so it acts on every state for the
whole flight, while the gust only bites during the crossing. Ungated noise buys
more boundary blur per newton than gated noise does.

**quad2d `baseline` sits at K=1, not K=20.** Zero force with no ambient means
there is no noise at all and every flight is byte-identical, so extra trials
would redraw the same rollout. Its `trials` array is all ones by design, and it
is excluded from every top-up.

## The reachability shortcut is unsound with an ungated ambient term

The quad2d collector settled a start in one flight whenever the undisturbed path
never came within `MARGIN` of the band, reasoning that a gated disturbance cannot
reach it. True of the corridor, which `sigma` gates. **False of the ambient
term**, which `build()` adds as ungated `white_noise` acting on every state for
the whole rollout.

Measured 2026-08-19 at f_max 0.08 / ambient 0.06: of 10 shortcut states re-flown
20 times, **7 varied**. They had been recorded as `p_success = 1.0` and came back
13, 15, 17, 17, 17, 17 and 18 of 20.

Both collectors now gate the shortcut on `ambient_on`. quad3d had the same trap
available, since its curtain is likewise gated, and was written with the gate
from the start rather than retrofitted. The quad2d spec carries a `FALSIFIED
2026-08-19` block; the affected shards were repaired in place with a `k1-20`
window rather than recollected, which is why quad2d shards have one more window
than quad3d's at every K.

## quad3d runs different deadlines on its two splits

Eval goes through `q3_corridor_common.roll()` at that module's `HORIZON = 2000`
(20 s, the memo-D deadline). Train goes through
`generate_quadrotor_3d_noisy.run()`, hardcoded to **that** module's
`HORIZON = 1000` (10 s, what the older `noisy_dynamics` family was labelled at).
So train labels sit at a stricter deadline than eval labels.

Measured on the shipped data 2026-08-20: the train deadline truncated **1 of
800,000** trajectories at `f_max 0.25` and **0** at `0.30`. Recorded rather than
recollected. Changing it means recollecting 800k trajectories, so do not flip it
silently mid-family. quad2d has no such split — both its paths use `roll()` at
`HORIZON = 1200`.

## k-windows: how a collection is topped up without recollecting

`rollout_seed(base, split_id, index, trial)` is a pure function of its
coordinates, so trial `k` draws the same noise no matter when it runs. That makes
a trial count extendable after the fact. A **window** is a shard file covering
trials `[trial_lo, trials)`, written by
`--trial_lo <lo> --trials <hi>`. The reducer sums `hits` and `trials_used` across
every window it finds for a shard index, recovering exactly what one
uninterrupted run at the higher K would have produced.

Two constraints, both able to corrupt a dataset silently if broken:

- **`--nshards` must match the original.** The reducer merges by shard index and
  asserts the starts agree. A different split does not merge; it fails, but only
  after the whole campaign has burned.
- **Never run a top-up while jobs for the same shards are live.** The collector
  skips an `--out` that already exists, but a shard still in flight has not
  written its file yet, so two processes race on one `np.savez` path.

Seeds cannot collide across windows. A collision needs
`(k1 - k2) * 104729 = 0 mod (2**31 - 1)`, and that modulus is a Mersenne prime,
so it requires `|k1 - k2| >= 2**31 - 1`. Verified by brute force over 200,000
states: zero cases where the 100 seeds were not all distinct.

**The trial-0 probe is dead work in a top-up window with an ambient term.** Two
readers consume it: the first window records its hit, and the shortcut needs
`entered`. A top-up with ambient on has neither, so the collector now skips it,
saving one rollout per state. Verified behaviour-neutral by running a 5-state
`k20-23` window through the old and new code: identical `hits`, `trials_used`,
`starts`, `det_labels`.

## What more trials actually buy

The eval set was carried from K=20 to K=50 to K=100. Interior fraction, the share
of states that sometimes reach the goal and sometimes do not:

| config | K=20 | K=50 | K=100 |
| --- | --- | --- | --- |
| quad2d `sharp` | 7.01% | 8.39% | 9.29% |
| quad2d `smooth` | 8.96% | 10.63% | 11.55% |
| quad3d `f_0.25` | 13.46% | 15.37% | 16.33% |
| quad3d `f_0.30` | 15.87% | 17.82% | 18.72% |

It only ever rises, and that is the acceptance check: more flights can expose a
state that looked certain and is not, never the reverse. A drop means the window
merge is wrong.

Returns halve at each doubling. On `f_0.25`, 20 to 50 bought 1.91 points and 50
to 100 bought 0.96. The mechanism is detection, not precision: a state with true
`p = 0.99` survives 50 flights unexposed 61% of the time and 100 flights 37% of
the time. Extrapolating, K=200 buys roughly half a point more. K=100 is where
this stopped.

## Acceptance checks a corridor collection must pass

Every one of these has caught something real here.

1. `mean_trials` exactly equal to K, and `trials_shortcut` zero. Catches states
   settled on a single flight.
2. Every top-up window carrying exactly `hi - lo` new trials per state, with no
   spread.
3. Start states in a new window matching the window it will be summed with.
   A mismatch merges into a quietly wrong dataset instead of erroring.
4. `p_success == hits / trials`, `hits <= trials`, offsets strictly increasing
   and spanning the state array, `cal_set + test_set` summing to the eval rows.
5. Column counts per system. quad2d is 6-D throughout. quad3d stores 13-D
   quaternion rows for trajectory states and eval starts, but **12-D Euler rows**
   for train starts, because `sampler_starts()` emits the sampler's own layout
   and it is kept verbatim so index `i` still matches the shipped deterministic
   set. Both reducers assert this per shard.
6. Expected-vs-actual shard count after the queue drains. Amarel's `main`
   partition is preemptible: a K=50-to-100 quad2d campaign lost 7 jobs and 184
   shards to preemption, in contiguous runs of 32 because that is one job's
   block. An empty queue is not completion.

Related: [datasets.md](datasets.md) for the families this one sits beside and the
publication layout it shares, [architecture.md](architecture.md) for the
`disturbances` mechanism and the `dynamics` channel it uses,
[glossary.md](glossary.md) for interior fraction, entry-cut and matchedness,
[compute.md](compute.md) for preemption and shard-count checking.
