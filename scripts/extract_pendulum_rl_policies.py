"""Extract the inverted-pendulum SAC policies + generate golden test fixtures.

This is a ONE-TIME oracle-generation tool. It must be run with the
``inverted_pendulum`` conda env's Python (Python 3.12, stable-baselines3 2.9.0,
torch 2.5.1), because that is the only place the trained SAC ``.zip`` models
load. It reads from the source system and writes into this (safe-control-gym)
repo:

  * ``safe_control_gym/controllers/pendulum_rl/models/{variant}_{strength}.npz``
    -- the SAC actor MLP weights, in a plain, version-agnostic NumPy format. This
    is a transient cross-env handoff: run ``scripts/convert_pendulum_models_to_pt.py``
    (in the safe-control-gym env) to turn each ``.npz`` into the native torch
    ``.pt`` state-dict the ``pendulum_rl`` controller actually loads. (The ``.pt``
    must be written by safe-control-gym's own torch, so it cannot be produced
    here directly.)
  * ``.../models/manifest.json`` -- provenance (source zip, checkpoint step,
    versions).
  * ``tests/test_inverted_pendulum/fixtures/*.json`` -- golden fixtures the
    safe-control-gym port is tested against:
      - ``env_rollouts.json``  : (x0, action seq) -> state seq from the IP env.
      - ``lqr_gain.json``      : the IP LQR gain K and a few (state -> u) pairs.
      - ``rl_golden.json``     : per-model (theta, thetadot) -> action pairs.

Run:  /common/users/shared/pracsys/st1122/inverted_pendulum/.env/bin/python \
          scripts/extract_pendulum_rl_policies.py

The deterministic SAC actor forward reproduced downstream is:
    h = [cos th, sin th, thdot / thdot_max]
    for (W, b) in hidden layers: h = relu(W @ h + b)     # net_arch [256, 256]
    mean = W_mu @ h + b_mu
    action = u_sat * tanh(mean)                           # squash + symmetric unscale
"""

import json
import math
import os

import numpy as np

IP_ROOT = "/common/users/shared/pracsys/st1122/inverted_pendulum"
SCG_ROOT = "/common/home/st1122/Projects/safe-control-gym"

MODELS_DIR = os.path.join(SCG_ROOT, "safe_control_gym/controllers/pendulum_rl/models")
FIXTURES_DIR = os.path.join(SCG_ROOT, "tests/test_inverted_pendulum/fixtures")

# seed-s0 strong (model_best.zip) + weak (cusp checkpoint resolved once via
# scripts/pick_checkpoint.py, cusp mode -- the strongest genuinely sub-converged
# snapshot before success crosses the ~0.9 convergence threshold).
WEAK_STEP = {"v1": 90000, "v2": 60000, "v3": 70000, "v4": 60000}
VARIANTS = ["v1", "v2", "v3", "v4"]

# torch float32 matmul and numpy float32 matmul reduce in different orders, so
# the reproduced action can differ from SB3's by ~1e-6; require agreement well
# below any control-relevant scale.
FWD_TOL = 1e-5


def model_path(variant, strength):
    run = os.path.join(IP_ROOT, "runs2x2", f"{variant}_s0")
    if strength == "strong":
        return os.path.join(run, "model_best.zip")
    return os.path.join(run, "checkpoints", f"step_{WEAK_STEP[variant]}.zip")


def numpy_forward(layers, mu_w, mu_b, u_sat, obs):
    """Deterministic SAC actor forward in float32 -> physical action."""
    h = np.asarray(obs, dtype=np.float32)
    for w, b in layers:
        h = np.maximum(0.0, w @ h + b)
    mean = mu_w @ h + mu_b
    return u_sat * np.tanh(mean)


def extract_one(variant, strength, u_sat, theta_dot_max, action_repeat):
    from stable_baselines3 import SAC

    path = model_path(variant, strength)
    model = SAC.load(path, device="cpu")
    sd = model.policy.actor.state_dict()

    # latent_pi is Sequential(Linear, ReLU, Linear, ReLU, ...); pull each Linear.
    layer_idx = sorted(
        {int(k.split(".")[1]) for k in sd if k.startswith("latent_pi.")}
    )
    layers = []
    npz = {}
    for i, li in enumerate(layer_idx):
        w = sd[f"latent_pi.{li}.weight"].cpu().numpy().astype(np.float32)
        b = sd[f"latent_pi.{li}.bias"].cpu().numpy().astype(np.float32)
        layers.append((w, b))
        npz[f"hidden_{i}_weight"] = w
        npz[f"hidden_{i}_bias"] = b
    mu_w = sd["mu.weight"].cpu().numpy().astype(np.float32)
    mu_b = sd["mu.bias"].cpu().numpy().astype(np.float32)
    npz["mu_weight"] = mu_w
    npz["mu_bias"] = mu_b
    npz["n_hidden"] = np.int64(len(layers))
    npz["u_sat"] = np.float64(u_sat)
    npz["theta_dot_max"] = np.float64(theta_dot_max)
    npz["action_repeat"] = np.int64(action_repeat)
    npz["activation"] = np.array("relu")

    # Validate the NumPy forward reproduces SB3 predict(deterministic=True).
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(200):
        obs = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        sb3_a, _ = model.predict(obs, deterministic=True)
        my_a = numpy_forward(layers, mu_w, mu_b, u_sat, obs)
        max_err = max(max_err, float(np.abs(sb3_a.reshape(-1) - my_a.reshape(-1)).max()))
    assert max_err <= FWD_TOL, f"{variant}/{strength}: forward mismatch {max_err:.2e} > {FWD_TOL:.0e}"

    out = os.path.join(MODELS_DIR, f"{variant}_{strength}.npz")
    np.savez(out, **npz)

    # Golden (theta, thetadot) -> action pairs for the safe-control-gym test.
    golden = []
    states = [
        [math.pi, 0.0], [0.0, 0.0], [1.5, -2.0], [-2.0, 3.0],
        [0.5, 0.5], [-1.0, -1.0], [2.5, 5.0], [-2.5, -5.0],
    ]
    for th, thd in states:
        obs = np.array([math.cos(th), math.sin(th), thd / theta_dot_max], dtype=np.float32)
        a, _ = model.predict(obs, deterministic=True)
        golden.append({"theta": th, "thetadot": thd, "action": float(np.asarray(a).reshape(-1)[0])})

    from stable_baselines3 import __version__ as sb3_version
    import torch
    info = {
        "variant": variant, "strength": strength,
        "source_zip": path,
        "checkpoint_step": (None if strength == "strong" else WEAK_STEP[variant]),
        "sb3_version": sb3_version, "torch_version": torch.__version__,
        "net_arch": [int(w.shape[0]) for w, _ in layers],
        "forward_max_err": max_err,
    }
    return info, golden


def gen_env_rollouts(u_sat):
    from inverted_pendulum import InvertedPendulumEnv
    from inverted_pendulum.params import PendulumParams

    params = PendulumParams(max_steps=1000)
    scenarios = []
    rng = np.random.default_rng(7)
    specs = [
        ("freeswing_zero", [3.0, 0.0], [0.0] * 60),
        ("spinup_pos_clip", [math.pi, 0.0], [u_sat] * 90),
        ("spinup_neg_clip", [-3.0, 0.0], [-u_sat] * 90),
        ("random_torque", [1.5, -2.0], list(rng.uniform(-u_sat, u_sat, size=70))),
        # Start with thetadot just under the bound and push -- forces the
        # theta_dot clip to engage and be ridden for many steps.
        ("clip_pos_ride", [1.0, 6.2], [u_sat] * 40),
        ("clip_neg_ride", [-1.0, -6.2], [-u_sat] * 40),
    ]
    for name, x0, actions in specs:
        env = InvertedPendulumEnv(params=params)
        env.reset(options={"state": np.array(x0, dtype=np.float64)})
        states, terminated = [], False
        for u in actions:
            _, _, term, trunc, info = env.step(np.array([u], dtype=np.float64))
            states.append([float(info["true_state"][0]), float(info["true_state"][1])])
            if term or trunc:
                terminated = bool(term)
                break
        scenarios.append({
            "name": name, "x0": [float(v) for v in x0],
            "actions": [float(u) for u in actions[: len(states)]],
            "states": states, "terminated": terminated,
        })
    return scenarios


def gen_lqr_golden():
    from inverted_pendulum.lqr import LQRController
    from inverted_pendulum.params import PendulumParams

    params = PendulumParams()
    lqr = LQRController(params=params)
    pairs = []
    for th, thd in [[0.05, 0.0], [0.0, 0.3], [-0.1, 0.2], [0.2, -0.5]]:
        u = float(lqr(np.array([th, thd], dtype=np.float64))[0])
        pairs.append({"theta": th, "thetadot": thd, "action": u})
    return {"K": [float(v) for v in np.asarray(lqr.K).reshape(-1)],
            "u_sat": float(params.u_sat), "pairs": pairs}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    from inverted_pendulum.params import PendulumParams
    p = PendulumParams()
    u_sat, theta_dot_max, action_repeat = p.u_sat, p.theta_dot_max, 4

    manifest, rl_golden = [], {}
    for variant in VARIANTS:
        for strength in ("strong", "weak"):
            info, golden = extract_one(variant, strength, u_sat, theta_dot_max, action_repeat)
            manifest.append(info)
            rl_golden[f"{variant}_{strength}"] = golden
            print(f"[ok] {variant}/{strength}  err={info['forward_max_err']:.2e}  "
                  f"arch={info['net_arch']}  <- {os.path.basename(info['source_zip'])}")

    with open(os.path.join(MODELS_DIR, "manifest.json"), "w") as f:
        json.dump({"u_sat": float(u_sat), "theta_dot_max": float(theta_dot_max),
                   "action_repeat": action_repeat, "obs_transform": "[cos th, sin th, thdot/thdot_max]",
                   "models": manifest}, f, indent=2)

    env_rollouts = gen_env_rollouts(u_sat)
    with open(os.path.join(FIXTURES_DIR, "env_rollouts.json"), "w") as f:
        json.dump({"params": {"g": p.g, "l": p.l, "m": p.m, "b": p.b, "dt": p.dt,
                              "u_sat": p.u_sat, "theta_dot_max": p.theta_dot_max,
                              "goal_threshold": p.goal_threshold},
                   "scenarios": env_rollouts}, f, indent=2)
    for s in env_rollouts:
        print(f"[env] {s['name']:16s} steps={len(s['states'])} terminated={s['terminated']}")

    with open(os.path.join(FIXTURES_DIR, "lqr_gain.json"), "w") as f:
        json.dump(gen_lqr_golden(), f, indent=2)
    with open(os.path.join(FIXTURES_DIR, "rl_golden.json"), "w") as f:
        json.dump(rl_golden, f, indent=2)

    print(f"\nWrote models -> {MODELS_DIR}")
    print(f"Wrote fixtures -> {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
