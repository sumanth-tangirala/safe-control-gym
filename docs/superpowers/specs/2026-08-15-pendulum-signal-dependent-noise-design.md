# Pendulum signal-dependent actuator noise

**Date:** 2026-08-15
**Status:** design
**Scope:** `safe_control_gym/envs/disturbances.py`,
`generate_inverted_pendulum_trajectories.py`,
`.claude/docs/{datasets,architecture,glossary}.md`

A pendulum stochastic family whose noise scale grows with the commanded torque,
rather than being a constant of the level. Same LQR, same grid, same horizon and
success rule as `stochastic/pendulum/noisy_torque/`, so the two are comparable
cell-for-cell. The only thing that changes is *where in the state space the
noise lives*.

## The noise

```
xdot = f(x, clip(u + w, +/- u_sat))          u_sat = 0.6371781908344007

w ~ Normal(0, alpha + beta * |u|)            alpha = 0.008
```

`|u|`, not `u`: with a signed command the scale parameter goes negative at
`u = -0.637` (`0.008 - 0.0255 = -0.0175`), which is not a standard deviation.

`alpha + beta*|u|` is the **standard deviation**, not the variance. The two
differ by ~5x here and the distinction is load-bearing: as a std the family is
weaker than the existing `tau = 0.10` level; as a variance it would sit in the
middle of that sweep.

Clipping is unchanged from the `tau` family — noise is added to the command and
the **sum** is clipped at saturation, so `w` itself is unbounded (Gaussian) while
the applied torque is not.

## Why this is a different mechanism, not another magnitude

`alpha` and `beta` control separate things:

- **`alpha` is the noise floor.** It is the sigma that survives as `u -> 0`, i.e.
  at the goal, where the LQR commands almost nothing. It sets whether the settled
  region fits inside the success box.
- **`beta` is the effort-proportional term.** It only bites while the controller
  is working hard, far from upright. It disrupts the *transient*, not the hold.

The `tau` family cannot separate these — its sigma is constant everywhere,
including at the goal, which is what creates the noise floor recorded in
`glossary.md` (settled region `|theta_dot| <= 0.4031*tau`).

Under this family the floor is set by `alpha` alone. At `alpha = 0.008` the
equivalent uniform level is `tau = alpha*sqrt(3) = 0.0139`, giving a settled
`|theta_dot| <~ 0.006` against a 0.05 box — roughly 9x margin. So the goal stays
comfortably reachable no matter how hard `beta` is pushed, which is exactly the
property the fixed-sigma families lack.

**Predicted consequence, to be checked rather than assumed:** at matched average
sigma this family should retain substantially more success than the `tau` family,
because the noise goes quiet precisely where holding the goal is hard.

## Levels

`alpha` fixed at 0.008; sweep `beta` only. An `alpha` sweep (which would
straddle the ~0.07 point where the floor starts eating the box) is deliberately
out of scope here.

| level | alpha | beta | sigma at u=0 | sigma at |u|=u_sat |
| --- | --- | --- | --- | --- |
| baseline | 0 | 0 | 0 | 0 |
| beta_0.04 | 0.008 | 0.04 | 0.0080 | 0.0335 |
| further levels | 0.008 | TBM | 0.0080 | TBM |

Only the first two are fixed. The remaining levels come from a sweep, as for
every other family here — levels are **not** chosen a priori, because they are
coupled to the success rule and the horizon and do not transfer across either
(measured on quad3d: retention 0.618 under a 0.1 box vs 0.015 under a 0.05 ball
at the same force).

Expected from `beta = 0.04`: sigma tops out at 0.0335, equivalent to `tau ~ 0.058`
in std terms — weaker than the existing `tau = 0.10` level, which measured
p = 0.3513 against a deterministic 0.3860. So this level should land near
**p ~ 0.375-0.38** with an interior fraction of a few percent, i.e. nearly
deterministic. The sweep exists to find where `beta` actually bites.

## Implementation

A new `Disturbance` subclass, registered in `DISTURBANCE_TYPES` alongside the
existing five. This belongs in the library, not the collector: it is a
disturbance model exactly like `UniformNoise` and `WhiteNoise`, not collection
policy.

`WhiteNoise` cannot be reused — its `std` is fixed at construction.
`StateDependentDisturbance` exists in the file but is not registered and reads
the *state*, not the commanded signal.

```python
class SignalDependentNoise(Disturbance):
    def apply(self, target, env):
        std = self.alpha + self.beta * np.abs(target)
        return target + self.np_random.normal(0, std, size=self.dim)
```

`apply(target, env)` already receives the action as `target` on the `action`
channel, so no change to the env or to `_preprocess_control` is needed. The
disturbance is applied before the `u_sat` clip, as all action-channel
disturbances here are.

## Layout

```
DATA_ROOT/stochastic/pendulum/signal_dependent/lqr/beta_<b>/
```

A sibling of `noisy_torque/`, not a level inside it — the mechanisms share no
parameterisation, and per the house rule a different mechanism gets its own
directory rather than being filed as another level.

Everything else matches `noisy_torque/`: the 49,770-cell eval grid, horizon 800,
`K = 100`, per-channel box `|theta| < 0.05 and |theta_dot| < 0.05` with
entry-cut and no dwell, `rollout_seed` excluding the level so levels are paired
under common random numbers, and the same npz + `eval_states.txt` layout.

## Verification

1. **Baseline gate.** `alpha = beta = 0` must reproduce the deterministic labels
   on the eval grid. This is the same-code baseline every family here carries;
   the shipped `tau_0.00` is not it, because it was collected at a different
   horizon (1000, in `lqr_legacy_20260806/`).
2. **Sigma check.** Draw `w` at fixed `u` and confirm the empirical std matches
   `alpha + beta*|u|` — cheap, and it catches a std/variance mix-up directly.
3. **Floor check.** Measure the settled region at the largest `beta` and confirm
   it stays inside the 0.05 box, i.e. that the family really has no noise floor.
4. **Comparison.** Report this family against `noisy_torque` at matched average
   sigma on the same cells. That comparison is the reason the family exists.

## Rejected

**A single multiplier `k` on both constants.** Convenient — it mirrors the 1-D
`tau` sweep and keeps the level count at five — but it moves the floor and the
effort term together, so a drop in `p` could not be attributed to either. That is
the same confound this session removed from the cartpole datasets, and it is not
worth reintroducing for tidiness.

**A full `alpha` x `beta` factorial.** 16 levels for perhaps 2x the information
over a one-at-a-time design. Out of scope now; the `alpha` axis is the more
interesting of the two and deserves its own spec.
