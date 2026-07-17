# Inverted Pendulum Integration — Design

**Date:** 2026-07-16
**Status:** Approved (design phase)

## Goal

Integrate the standalone inverted-pendulum system from
`/common/users/shared/pracsys/st1122/inverted_pendulum` into safe-control-gym as a
first-class environment plus a set of controllers, and add a trajectory-generation
script consistent with the existing cartpole/quadrotor generators.

Deliverables (confirmed with user):

1. A registered `inverted_pendulum` environment (`BenchmarkEnv` subclass).
2. Controllers:
   - `pendulum_lqr` — the inverted pendulum's own bounds-normalized LQR, ported verbatim.
   - `pendulum_rl` — the trained SAC swing-up policies, as **standalone** controllers
     (no LQR handoff), covering **V1–V4 × {strong, weak}** for seed **s0** = 8 policies.
3. `generate_inverted_pendulum_trajectories.py` — writes `sequence_*.txt` datasets like
   `generate_cartpole_trajectories.py`.

Out of scope: the RL→LQR `SwitchingController` composite, the `CautiousLQRController`,
noise models, seeds s1/s2, and the V3/V4 ellipse "flagship" configs (`runs_final`).

## Source system summary

- **Env** (`inverted_pendulum/env.py`): state `[θ, θ̇]`, dynamics
  `θ̈ = (g/l)·sin θ + u/I − (b/I)·θ̇` with `I = m·l²`, **explicit Euler at dt=0.01**;
  after each step θ is wrapped to `[−π, π]` and θ̇ is clipped to `±θ̇_max`; u is clipped
  to `±u_sat`. `θ=0` is the upright unstable equilibrium.
- **Params** (`inverted_pendulum/params.py`): `g=9.81, l=0.5, m=0.15, b=0.1, dt=0.01,`
  `u_sat=0.6371781908344007, θ̇_max=2π, goal=(0,0), goal_threshold=0.075, max_steps=1000`.
- **LQR** (`inverted_pendulum/lqr.py`): analytic linearization at upright, then
  **bounds-normalized** by `Tx=diag(π, θ̇_max)`, `Tu=u_sat` before solving the continuous
  ARE. Gain `K` shape (2,); `u = clip(−K·(x−goal), ±u_sat)`.
- **RL** (`inverted_pendulum/rl/`): SB3 **SAC** `MlpPolicy`, obs transform
  `[θ,θ̇] → [cos θ, sin θ, θ̇/θ̇_max]`, action-repeat 4, deterministic eval. V1–V4 differ
  **only** in the sparse goal region used during training; the policy weights are what
  we port. Trained with SB3 2.9.0 / torch 2.5.1 / gymnasium 1.3.0 / numpy 2.5 / Py3.12.
- **Model locations** (seed s0, `runs2x2/`):
  - strong = `runs2x2/{v1,v2,v3,v4}_s0/model_best.zip`
  - weak (resolved cusp checkpoints via `scripts/pick_checkpoint.py`):
    - v1 → `checkpoints/step_90000.zip` (train success 0.025)
    - v2 → `checkpoints/step_60000.zip` (0.494)
    - v3 → `checkpoints/step_70000.zip` (0.593)
    - v4 → `checkpoints/step_60000.zip` (0.346)

## Target system summary (safe-control-gym)

- Envs/controllers are registered centrally in
  `safe_control_gym/envs/__init__.py` and `safe_control_gym/controllers/__init__.py`
  (not at file bottom).
- Env is a `BenchmarkEnv(gym.Env, ABC)` subclass. Cartpole uses PyBullet, but the base
  class does **not** require it; the pendulum will integrate its ODE directly.
- `step()` returns the **old 4-tuple** `(obs, reward, done, info)`.
- Controllers subclass `BaseController`, receive the env as a **factory** `env_func`
  (not an instance), and must implement `select_action(obs, info)`, `reset`, `close`.
- The `scg` conda env is Python 3.10 / torch 1.13.1 / gymnasium 0.28.1, and has **no
  stable-baselines3**. Installing SB3 2.9.0 would force upgrading torch/gymnasium/numpy
  and likely break safe-control-gym → **SB3 must not become a runtime dependency.**

## Key design decisions

### D1 — RL: extract weights, no SB3 runtime dependency
A **one-time extraction** (run in the `inverted_pendulum` conda env, where SB3 loads)
pulls each SAC actor's MLP into a version-agnostic `.npz`. The `scg` `pendulum_rl`
controller does a **pure-NumPy deterministic forward pass** reproducing
`model.predict(deterministic=True)`:

```
h = [cos θ, sin θ, θ̇/θ̇_max]
for (W, b) in hidden layers: h = relu(W·h + b)     # net_arch [256,256], ReLU
mean = W_mu·h + b_mu
action = u_sat · tanh(mean)                          # symmetric-bounds unscale of squashed output
u = clip(action, ±u_sat)
```

Extraction **validates** the NumPy forward against SB3's `predict` on random obs
(≤1e-6) before writing, so the port is faithful by construction. Committed artifacts
shrink from ~24 MB of zips to ~2 MB of npz.

*Rejected:* (a) installing SB3 into `scg` — breaks the pinned stack; (b) loading the raw
torch-2.5 `.pth` under torch 1.13 — cross-version pickle risk.

### D2 — Env: native `BenchmarkEnv`, exact IP physics
A proper registered `inverted_pendulum` env with a real CasADi symbolic model (so it is
a first-class citizen and scg's generic tools work), but `step()` integrates the IP
dynamics faithfully: **explicit Euler at dt=0.01, θ wrap, θ̇ clip, u clip**. No PyBullet.

*Rejected:* scg's usual symbolic RK4 integrator — would drift from the physics the
policies were trained under.

### D3 — LQR: port the bounds-normalized LQR verbatim
`pendulum_lqr` reproduces `_normalized_linearization` + continuous ARE, so its gain `K`
matches the source exactly (the ROA the policies were trained against depends on it).

*Rejected:* reusing scg's generic symbolic `lqr` — different gain, shifted ROA.

### D4 — Action repeat lives in the RL controller
Env runs at `pyb_freq = ctrl_freq = 100` (one Euler substep per control step, dt=0.01),
so goal/wrap/clip are checked every dt — matching the source. Action-repeat-4 is handled
**inside `pendulum_rl`** (re-query the MLP every 4th call, hold the action). This keeps a
single env config usable by every controller (LQR queried every step).

### D5 — θ̇ clipping, not termination (approved divergence)
The pendulum **clips** θ̇ at its bound and **wraps** θ; there is **no out-of-bounds
failure**. A trajectory ends only on **goal-reached (success)** or **timeout**. This
diverges from the cartpole/quadrotor "termination = bound" convention in the repo's
`CLAUDE.md`, and is intentional: the trained policies were trained under clipping, and
the state space stays closed via projection. Explicitly approved by the user.

## Component specifications

### Environment — `safe_control_gym/envs/gym_control/inverted_pendulum.py`
`class InvertedPendulum(BenchmarkEnv)`:
- `NAME = 'inverted_pendulum'`.
- State `[θ, θ̇]` (nx=2), action `[u]` (nu=1). `STATE_LABELS=['theta','theta_dot']`,
  `STATE_UNITS=['rad','rad/s']`, `ACTION_LABELS=['U']`, `ACTION_UNITS=['N·m']`.
- `__init__`: custom pendulum params (with `PendulumParams` defaults) set first, then
  `super().__init__(...)`. Build `X_GOAL=[0,0]`, `U_GOAL=[0]`, then `_setup_symbolic()`.
- `_setup_symbolic`: CasADi `X=[θ, θ̇]`, `U=[u]`,
  `X_dot=[θ̇, (g/l)·sin θ + u/I − (b/I)·θ̇]`, `Y=X`, quadratic cost, `params` with
  `X_EQ=[0,0]`, `U_EQ=[0]`; `self.symbolic = SymbolicModel(...)`.
- `_set_action_space`: Box `[−u_sat, u_sat]`; `physical_action_bounds`.
- `_set_observation_space`: `state_space` and `observation_space` over
  `θ∈[−π,π]`, `θ̇∈[−θ̇_max, θ̇_max]` (extended by goal horizon if `obs_goal_horizon>0`).
- `reset`: `before_reset(seed)`; set state from `init_state` (exact) or uniform sample
  over the state box; `after_reset`. Returns `(obs, info)`.
- `step`: `before_step(action)` → u; integrate `PYB_STEPS = pyb_freq//ctrl_freq` Euler
  substeps, each: `u=clip(u,±u_sat)`, Euler dt, wrap θ, clip θ̇, (optional dynamics
  noise hook), per-substep goal check with early break; then
  `_get_observation/_get_reward/_get_done/_get_info`; `after_step`; return 4-tuple.
- `_get_done`: `dist(state, goal) < goal_threshold` (success) — no out-of-bounds path.
- `normalize_action`/`denormalize_action`/`_preprocess_control`: physical action space
  (identity unless `NORMALIZED_RL_ACTION_SPACE`, mirroring cartpole's `action_scale`).
- `render`/`close`: minimal (no PyBullet).

`inverted_pendulum.yaml`: mirror `cartpole.yaml` keys — `ctrl_freq: 100`,
`pyb_freq: 100`, `episode_len_sec: 10`, `task: stabilization`, `cost`, `init_state`,
`randomized_init`, plus pendulum params so they are overridable.

Register in `safe_control_gym/envs/__init__.py`:
```python
register(idx='inverted_pendulum',
         entry_point='safe_control_gym.envs.gym_control.inverted_pendulum:InvertedPendulum',
         config_entry_point='safe_control_gym.envs.gym_control:inverted_pendulum.yaml')
```

### Controller — `safe_control_gym/controllers/pendulum_lqr/pendulum_lqr.py`
`class PendulumLQR(BaseController)`:
- `__init__(env_func, q_lqr=None, r_lqr=None, **kwargs)`: `self.env = env_func()`; read
  physical params from the env; build normalized `A,B` (`Tx=diag(π,θ̇_max)`, `Tu=u_sat`);
  `solve_continuous_are`; `self.K = (R⁻¹ Bₙᵀ S).ravel()`.
- `select_action(obs, info=None)`: `clip(−K·(obs − X_GOAL), ±u_sat)`.
- `reset`: `self.env.reset()`; `close`: `self.env.close()`.
- `pendulum_lqr.yaml`: `q_lqr: [1, 1]`, `r_lqr: [1]` (identity Q, R=[[1]] — the IP default).

### Controller — `safe_control_gym/controllers/pendulum_rl/pendulum_rl.py`
`class PendulumRL(BaseController)`:
- `__init__(env_func, model_path=None, action_repeat=4, **kwargs)`; `self.env = env_func()`.
- `load(path)`: read `.npz` → layer weights/biases, activation, net_arch, `θ̇_max`,
  `u_sat`, `action_repeat`.
- `obs_normalizer`: identity (`BaseNormalizer`) so the traj-gen `ctrl.obs_normalizer(obs)`
  contract holds; the real obs transform happens inside `select_action`.
- `select_action(obs, info=None)`: on repeat boundary, transform `[θ,θ̇]→[cos,sin,θ̇/max]`,
  NumPy MLP forward, `u = clip(u_sat·tanh(mean), ±u_sat)`; hold between re-queries.
- `reset`: reset repeat counter + env; `close`: `self.env.close()`.
- `pendulum_rl.yaml`: `model_path: null`, `action_repeat: 4`.
- Models: `models/{v1,v2,v3,v4}_{strong,weak}.npz` (~2 MB total) + `manifest.json`
  (source zip, checkpoint step, train success, SB3/torch versions).

Register both in `safe_control_gym/controllers/__init__.py`:
```python
register(idx='pendulum_lqr',
         entry_point='safe_control_gym.controllers.pendulum_lqr.pendulum_lqr:PendulumLQR',
         config_entry_point='safe_control_gym.controllers.pendulum_lqr:pendulum_lqr.yaml')
register(idx='pendulum_rl',
         entry_point='safe_control_gym.controllers.pendulum_rl.pendulum_rl:PendulumRL',
         config_entry_point='safe_control_gym.controllers.pendulum_rl:pendulum_rl.yaml')
```

### Extraction — `scripts/extract_pendulum_rl_policies.py`
Run once in the `inverted_pendulum` conda env. For each of the 8 s0 models:
`SAC.load(zip)` → extract `actor.latent_pi.*` + `actor.mu.*` + net_arch/activation;
**assert** NumPy forward matches `model.predict(obs, deterministic=True)` (≤1e-6) on a
batch of random obs; write `{v}_{strength}.npz` and a row in `manifest.json`; also emit
golden `(obs → action)` fixtures for the scg test suite. Emit golden env-rollout
fixtures and the golden LQR `K` here too (single source of truth for tests).

### Trajectory generation — `generate_inverted_pendulum_trajectories.py`
Repo root, structured like `generate_cartpole_trajectories.py`:
- CLI: `--controller {lqr,v1_strong,v1_weak,…,v4_weak}`, `--num_trajs`, `--random_init`,
  `--parallel`, `--seed`, `--output_dir`, `--save_freq`.
- Per trajectory: sample init `θ∈[−π,π]`, `θ̇∈[±θ̇_max]`; `make('inverted_pendulum', …)`;
  `make(controller_id, env_func, …)` (+ `ctrl.load(model_path)` for RL); roll to
  done/timeout.
- Output: `{output_dir}/trajectories/sequence_{idx}.txt` (`[θ, θ̇]` per line, comma-sep,
  6 decimals, θ wrapped), `roa_labels.txt` (per-trajectory success label),
  `dataset_description.json`.
- Default `--output_dir`:
  `…/genMoPlan/data_trajectories/inverted_pendulum_{controller}/` (distinct prefix so it
  will not collide with the source system's existing `pendulum_lqr_50k`, etc.).

## Testing plan

- **Env fidelity:** golden reference trajectories generated once from the IP env
  (committed fixtures); scg env reproduces to ≤1e-9 given identical init state + action
  sequence.
- **LQR gain:** scg `PendulumLQR.K` equals the golden IP `K`.
- **RL forward:** scg `PendulumRL.select_action` matches golden `(obs → action)` pairs.
- **Smoke:** `make('inverted_pendulum')` + each controller rolls a short trajectory;
  a `tests/test_inverted_pendulum/` suite following the `tests/test_examples` pattern.

## Risks / notes

- **Golden fixtures cross the Python-version boundary** (IP env 3.12 → scg 3.10). They
  are plain floats in a committed file, so no runtime interop is needed — the extraction
  script is the only thing that runs in the IP env.
- **SB3 actor internals** (`latent_pi`/`mu` names, squash + symmetric unscale) are
  assumptions verified by the ≤1e-6 validation gate in the extraction script; if SB3's
  layout differs, the gate fails loudly rather than shipping a wrong policy.
- **No out-of-bounds failure** (D5) means `roa_labels.txt` semantics for the pendulum are
  "reached goal within horizon" vs "timed out", not "in region of attraction / left
  bounds" as for cartpole. Documented in `dataset_description.json`.
