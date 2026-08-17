#!/usr/bin/env python3
'''Fit and evaluate the supervisory guard g(x): does controller 2 (LQR) alone
succeed from initial state x?

This is offline, tabular model-fitting -- no PyBullet, no rollouts. It reads
the already-regenerated baseline dataset (`--mode baseline` of
`generate_quadrotor_3d_composition.py`),
`quadrotor3D_lqr_regenerated/eval_states.txt` (40000 rows, 27 columns:
init(13), final(13), ctrl2_success(1)), uses the initial 13 columns as
features and ctrl2_success as the target, and reports accuracy/precision/
recall on a HELD-OUT split only -- never on the rows the model was fit on.

FEATURES are four physically-motivated scalars, all computable from a
dataset-order 13-dim state [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot,
p, q, r]:

    tilt      = rollout3d.tilt_from_quat_wxyz(quat)   -- rotation-matrix tilt,
                NEVER Euler angles (see rollout3d.py's module docstring,
                item 1, and CLAUDE.md).
    omega     = |[p, q, r]|                            body-rate magnitude
    speed     = |[x_dot, y_dot, z_dot]|                 linear speed
    dist      = |[x, y, z] - goal|                      distance from goal

These are exactly the scalars an LQR practitioner would reach for first: LQR
is a local linear controller, so its region of attraction is expected to
shrink as any of these grows. Angular position (tilt) is included; raw
position/attitude components are not, since a guard built from THESE four
rotation/translation-invariant scalars generalises across the direction of
the offset, not just its magnitude along particular axes.

CANDIDATES, cheapest first (spec: "prefer the simplest thing that works"):
  1. majority-class baseline (floor -- always predict the majority label)
  2. single-threshold rule on `dist` alone
  3. shallow decision tree (max_depth=3) over all four features -- fit with
     sklearn for convenience, but the CHOSEN rule is extracted as explicit
     thresholds and hardcoded into quad_composition/guard3d.py, so the
     rollout path never depends on sklearn at runtime.
  4. logistic regression over standardized features -- also extracted as
     explicit weights and hardcoded, same reason.

All candidates are fit on the TRAIN split only; every reported metric is
computed on the TEST split only.
'''

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

from quad_composition.rollout3d import GOAL_STATE, QUAT_SLICE, RATE_SLICE, tilt_from_quat_wxyz

EVAL_STATES = ('/common/users/shared/pracsys/genMoPlan/data_trajectories/'
              'deterministic/quadrotor3D_lqr_regenerated/eval_states.txt')

FEATURE_NAMES = ['tilt_rad', 'omega_norm', 'speed', 'dist_from_goal']


def compute_features(states):
    '''(n, 4) feature matrix [tilt, |omega|, speed, dist_from_goal] from an
    (n, 13) dataset-order state matrix. Vectorized except tilt, which goes
    through `tilt_from_quat_wxyz` (loops in Python -- pybullet's quaternion
    routine is not vectorized) row by row; 40000 rows takes well under a
    second.
    '''
    states = np.atleast_2d(np.asarray(states, dtype=float))
    tilt = np.array([tilt_from_quat_wxyz(s[QUAT_SLICE]) for s in states])
    omega = np.linalg.norm(states[:, RATE_SLICE], axis=1)
    speed = np.linalg.norm(states[:, 7:10], axis=1)
    dist = np.linalg.norm(states[:, 0:3] - np.asarray(GOAL_STATE[0:3]), axis=1)
    return np.column_stack([tilt, omega, speed, dist])


def load_dataset(path=EVAL_STATES):
    rows = np.loadtxt(path, delimiter=',')
    init = rows[:, :13]
    label = rows[:, 26].astype(int)
    return init, label


def split_indices(n, test_frac=0.2, seed=42):
    '''A fixed, reproducible train/test partition -- same seed every run, so
    "held out" means the same rows every time this script is invoked.
    '''
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    return idx[n_test:], idx[:n_test]


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    n = len(y_true)
    return {
        'n': n, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'accuracy': (tp + tn) / n if n else float('nan'),
        'precision': tp / (tp + fp) if (tp + fp) else float('nan'),
        'recall': tp / (tp + fn) if (tp + fn) else float('nan'),
    }


# ---------------------------------------------------------------------------
# Candidate 1: majority-class floor.
# ---------------------------------------------------------------------------

def fit_majority(y_train):
    majority = int(round(y_train.mean())) if y_train.mean() != 0.5 else 0
    return majority


def predict_majority(majority, n):
    return np.full(n, majority, dtype=int)


# ---------------------------------------------------------------------------
# Candidate 2: single threshold on `dist_from_goal` alone -- swept on TRAIN
# only, over the train distribution's own values (so the threshold is always
# a value the data actually takes).
# ---------------------------------------------------------------------------

def fit_single_threshold(feature_train, y_train):
    candidates = np.unique(feature_train)
    best_thresh, best_acc = candidates[0], -1.0
    for t in candidates:
        pred = (feature_train <= t).astype(int)
        acc = (pred == y_train).mean()
        if acc > best_acc:
            best_acc, best_thresh = acc, t
    return best_thresh


def predict_single_threshold(thresh, feature):
    return (feature <= thresh).astype(int)


# ---------------------------------------------------------------------------
# Candidate 3: shallow decision tree.
# ---------------------------------------------------------------------------

def fit_tree(X_train, y_train, max_depth=3, seed=42):
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    tree.fit(X_train, y_train)
    return tree


# ---------------------------------------------------------------------------
# Candidate 4: logistic regression on standardized features.
# ---------------------------------------------------------------------------

def fit_logreg(X_train, y_train, seed=42):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X_train - mean) / std
    clf = LogisticRegression(random_state=seed)
    clf.fit(Xs, y_train)
    return clf, mean, std


def predict_logreg(clf, mean, std, X):
    Xs = (X - mean) / std
    return clf.predict(Xs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval_states', default=EVAL_STATES)
    parser.add_argument('--test_frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tree_depth', type=int, default=3)
    parser.add_argument('--output', default=None, help='optional JSON report path')
    args = parser.parse_args(argv)

    init, label = load_dataset(args.eval_states)
    X = compute_features(init)
    train_idx, test_idx = split_indices(len(label), args.test_frac, args.seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = label[train_idx], label[test_idx]

    print(f'{len(label)} rows: {len(train_idx)} train / {len(test_idx)} test '
         f'(seed={args.seed}), base rate {label.mean() * 100:.2f}% ctrl2_success')
    print(f'features: {FEATURE_NAMES}')
    print()

    report = {'n_total': int(len(label)), 'n_train': int(len(train_idx)),
             'n_test': int(len(test_idx)), 'base_rate': float(label.mean()),
             'feature_names': FEATURE_NAMES, 'candidates': {}}

    # --- 1. majority ----------------------------------------------------
    majority = fit_majority(y_train)
    pred = predict_majority(majority, len(y_test))
    m = classification_metrics(y_test, pred)
    report['candidates']['majority'] = dict(m, majority_label=majority)
    print(f'1. majority-class floor (always predict {majority}):')
    print(f'   test accuracy={m["accuracy"]:.4f} precision={m["precision"]:.4f} '
         f'recall={m["recall"]:.4f}')
    print()

    # --- 2. single threshold on dist_from_goal ---------------------------
    dist_idx = FEATURE_NAMES.index('dist_from_goal')
    thresh = fit_single_threshold(X_train[:, dist_idx], y_train)
    pred = predict_single_threshold(thresh, X_test[:, dist_idx])
    m = classification_metrics(y_test, pred)
    report['candidates']['dist_threshold'] = dict(m, threshold=float(thresh))
    print(f'2. single threshold on dist_from_goal <= {thresh:.4f} m:')
    print(f'   test accuracy={m["accuracy"]:.4f} precision={m["precision"]:.4f} '
         f'recall={m["recall"]:.4f}')
    print()

    # --- 3. shallow decision tree ----------------------------------------
    tree = fit_tree(X_train, y_train, max_depth=args.tree_depth, seed=args.seed)
    pred = tree.predict(X_test)
    m = classification_metrics(y_test, pred)
    tree_text = export_text(tree, feature_names=FEATURE_NAMES)
    report['candidates']['tree'] = dict(m, max_depth=args.tree_depth, tree_text=tree_text,
                                        feature_importances=tree.feature_importances_.tolist())
    print(f'3. decision tree (max_depth={args.tree_depth}):')
    print(f'   test accuracy={m["accuracy"]:.4f} precision={m["precision"]:.4f} '
         f'recall={m["recall"]:.4f}')
    print(tree_text)
    print()

    # --- 4. logistic regression --------------------------------------------
    clf, mean, std = fit_logreg(X_train, y_train, seed=args.seed)
    pred = predict_logreg(clf, mean, std, X_test)
    m = classification_metrics(y_test, pred)
    coef = clf.coef_[0]
    report['candidates']['logreg'] = dict(
        m, mean=mean.tolist(), std=std.tolist(), coef=coef.tolist(),
        intercept=float(clf.intercept_[0]))
    print('4. logistic regression (standardized features):')
    print(f'   test accuracy={m["accuracy"]:.4f} precision={m["precision"]:.4f} '
         f'recall={m["recall"]:.4f}')
    for name, c in zip(FEATURE_NAMES, coef):
        print(f'   {name:<16} coef={c:+.4f}')
    print(f'   intercept={clf.intercept_[0]:+.4f}')
    print()

    if args.output:
        with open(args.output, 'w') as fh:
            json.dump(report, fh, indent=2)
        print(f'-> {args.output}')

    return report


if __name__ == '__main__':
    main()
