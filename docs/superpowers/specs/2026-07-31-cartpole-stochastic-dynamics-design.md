# Spec: Stochastic Dynamics for the Cartpole

Date: 2026-07-31
Scope: `configs/` (a new noise axis for cartpole), `safe_control_gym/envs/disturbances.py`
(seeding). No collector changes yet, no controller changes.

Status: **decision recorded, not implemented.** Nothing in `configs/` enables any
disturbance today; the cartpole is deterministic in every regime.

## Goal

Give the cartpole a stochastic dynamics axis so its collectors can produce the
same object the pendulum's already do — a per-start-state success probability
`p(success | x0, H)` — rather than a single deterministic success label.

The consumer is the downstream flow-matching terminal-state model, and the
question the datasets are meant to answer is the **ROA of a controller under
dynamics it does not model**.

## Decision

Use the existing `dynamics` disturbance mode, masked to the horizontal component,
with a **bounded** distribution:

```yaml
disturbances:
  dynamics:
  - disturbance_func: uniform
    mask: [1, 0]
    low: -<a>
    high: <a>
```

Magnitude `<a>` is deliberately left unset here — see "Open before implementation".

## Why this and not the alternatives

The decisive property is **matched vs unmatched**, not realism.

With `q = (x, theta)`, `l = EFFECTIVE_POLE_LENGTH = 0.5` (pivot-to-COM half length,
`cartpole.py:794`), the plant is `M(q)qdd + C + G = B*u` with `B = (1, 0)'`. That is
what `_setup_symbolic` encodes (`cartpole.py:463-465`).

An external world-frame force `w = (Fx, Fz)` at the pole COM enters as

```
J(theta) = [ 1       l*cos(theta) ]        Q = J(theta)' w
           [ 0      -l*sin(theta) ]

M(q)qdd + C + G = B*u + J(theta)' w
```

so the noise multiplies a **configuration-dependent** input map, not a constant.
Masked to `Fx`, `J'w = Fx * (1, l*cos(theta))`, whose component orthogonal to `B`
is `(0, l*Fx*cos(theta))` — a direct pivot torque, **maximal at upright**, which
is exactly where the stabilization task lives.

In `xdot = f(x, u) + g(x) w` form, with `x = (x, xdot, theta, thetadot)`:

```
g(x)   = ( 0, c_x(theta), 0, c_theta(theta) )'
c_theta = cos(theta) * M / ( m*(M+m)*D )
c_x     = 1/(M+m) - (m*l*cos(theta)/(M+m)) * c_theta
D       = l*( 4/3 - m*cos(theta)^2/(M+m) )
```

### Rejected: action noise (`disturbances.action`)

`xdot = f(x, u + w_a)`, so `g_a = df/du` and the perturbation lies in `range(B)` —
**matched**. The controller has authority in precisely that direction, so it
partially rejects the disturbance and the estimated ROA comes out biased toward
the nominal (noise-free) ROA. For a spec whose deliverable *is* the ROA, an
optimistic estimator is disqualifying.

Quantitatively, from the same solve: in the `thetaddot` numerator the coefficient
on `w_a` is `-cos(theta)/(M+m)` and on `Fx` is `+cos(theta)*M/(m*(M+m))`. Ratio
`M/m = 10`, opposite sign. One Newton at the pole COM is ten times the angular
effect of one Newton at the cart, and tips the pole the other way. **Noise
magnitudes are therefore not transferable between the two channels.**

Action noise keeps a role as a **training-time regulariser** — in a deterministic
sim, RL policies learn brittle bang-bang solutions that exploit exact simulator
behaviour. Train-time and eval-time noise being different is intentional here,
not an oversight.

### Rejected as the primary axis: domain randomisation + joint friction

The physically realistic account of "unmodeled dynamics" on a cartpole is, in
order of severity: rail and pivot friction; actuator dead zone, backlash and lag;
control latency; encoder quantisation; then parameter error. External stochastic
force is near the bottom of that list, and i.i.d. white force at 50 Hz models
none of it — real external disturbances are time-correlated.

Friction is also genuinely unmodeled: `_setup_symbolic` has no friction term, so
the controllers cannot compensate for it. Both mechanisms are currently disabled
on purpose — `cartpole.py:360` zeroes `linearDamping`/`angularDamping`, and
`cartpole.py:362` sets the default joint motor to `force=0`, which is PyBullet's
joint-friction mechanism.

This was rejected as the *primary* axis, not as wrong:

- Realism is the criterion for a sim-to-real claim. There is no target hardware,
  so it has no referent here.
- The existing factory is organised per-controller/per-noise-level, and the
  pendulum levels are process/actuation noise presets. Domain randomisation is a
  different axis, and adding it means changing the dataset layout rather than
  adding a level.
- Sampling structure differs. Under parameter randomisation the plant is fixed
  within an episode, so each draw traces a *deterministic* terminal point and
  `p(success|x0)` is an average of shifted indicator functions — a union of
  deterministic boundaries. Process noise gives a smooth field, a better-behaved
  target for the terminal-state model.

Keep it as a separately-named second axis if a robustness-to-model-error claim is
wanted later. It is the right tool for that claim and the wrong one for this
program.

### Rejected: `white_noise` (untruncated Gaussian)

With unbounded support no bounded set is invariant with probability 1 — escape is
almost sure over an infinite horizon — so there is no set-valued robust ROA, only
a finite-horizon probability decaying in `H`. Bounded noise is required if the
deliverable is a set. `DISTURBANCE_TYPES` (`disturbances.py:277`) offers `uniform`
as the only bounded i.i.d. option; there is no truncated Gaussian, which is why
the pendulum's `truncated_gaussian_act_*` presets have no cartpole counterpart.

### Rejected: porting the pendulum's state-additive noise

`pendulum_noise.py`'s dynamics models add `eps` directly to the state after
integration (`inverted_pendulum.py:173-177`), with the same sigma on theta (rad)
and thetadot (rad/s). Full rank, but no force generates it — it is a numerical
convenience ported for fidelity to the source system, not a physics claim. It is
also drawn every PyBullet substep, where the cartpole's `tab_force` is drawn once
per control step and held (`cartpole.py:592-600`), so the two diffuse differently
under `pyb_freq`.

Porting it would propagate a choice made for a reason that does not apply here.

### Rejected: leaving `Fz` unmasked

`Q_theta` from `Fz` scales as `-l*sin(theta)`, exactly zero at upright and at
hanging. For stabilisation it contributes nothing while adding a second magnitude
to calibrate. Revisit for swingup, where the pole spends time near horizontal.

## Cross-system comparability

Only the **action** channel is structurally comparable between cartpole and
pendulum: both additive on the physical command, both before the saturation clip,
both drawn once per control step. Note the parameterisation differs — pendulum
presets take *variance* (`act_noise_var`), `WhiteNoise` takes *std* — and the
authorities differ (`U_SAT_DEFAULT = 0.6371781908344007` N·m vs cartpole
`action_scale = 10.0` N).

The `dynamics` channels are **not** comparable, for the reasons above. Calibrate
each system's magnitude independently against its own success-rate curve; do not
carry a noise level across systems and do not imply comparability in the
directory naming.

## Open before implementation

1. **Seeding blocks reproducible collection.** `Disturbance.seed` binds
   `env.np_random` itself (`disturbances.py:33-35`), not a child stream. Enabling
   any disturbance shifts every other draw off that generator — init-state and
   inertial-prop randomisation included — and the noise sequence stops being a
   pure function of the `rollout_seed` coordinates. Resident invariant 3 requires
   a resumed run to draw exactly what an uninterrupted run would have. Fix with a
   per-disturbance `default_rng` derived from the rollout seed **before** any
   collection.

2. **Magnitude is uncalibrated.** No measurement supports a value yet. Sweep and
   pick from where the success rate starts to move. Note the `dynamics` channel
   bypasses the `physical_action_bounds` clip entirely, so nothing caps it — the
   bound must be chosen, not inherited.

3. **Noise intensity is tied to `ctrl_freq`.** `w_k` is drawn once per control
   step and held over the interval, so this is piecewise-constant random forcing,
   not Brownian: sigma is fixed independent of `dt`, and as `dt -> 0` the
   trajectory converges back to the deterministic ODE. The 50 Hz configs *define*
   the process. If control rate is ever swept, scale magnitude by
   `sqrt(dt_ref/dt)` or the comparison is meaningless.

4. **The ROA is now a field, not a set.** `p(success | x0, H)`, thresholded at
   some `alpha`. Both `alpha` and `H` are choices; report them.

## Honesty note

Every rejection above is analytic, not measured — this spec argues from the
equations of motion and from what the code does, and no rollouts were run. The
house rule prefers a measurement. The magnitude sweep in "Open before
implementation" is where the numbers should come from, and this spec should be
amended with them once they exist.

## Out of scope

Friction, actuator lag, control latency, encoder quantisation, and the
`randomized_inertial_prop` axis. All are worth having; none belongs in this
change.
