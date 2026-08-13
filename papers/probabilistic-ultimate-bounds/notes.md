---
uid: "8k1f3ypw"
title: "Continuous-time probabilistic ultimate bounds and invariant sets: Computation and assignment"
url: "https://www.sciencedirect.com/science/article/abs/pii/S0005109816301662"
tags: ["invariant-sets", "ultimate-bounds", "stochastic"]
verdict:
read: false
projects: "safe-control-gym"
---

How to COMPUTE the set I only sampled. My ultimate bound is 24 trajectories that did not leave a box; this is the machinery that turns that into a certificate. Directly extends compute_invariant_sets.py to the noisy case.

Revised 2026-08-06: I first thought bounded (uniform) noise made this unnecessary, since a hard minimal robust invariant set exists. Measured, that adversarial set is ~15x the observed settled region (|theta_dot| 3.11 vs 0.20 at tau=0.5) -- far too loose to call a goal. The probabilistic bound is the one worth having.
