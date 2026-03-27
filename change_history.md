# MC-PILOT Change History
AR525 Group-3, IIT Mandi

---

## Baseline Bugs Fixed

### Fix 1: T_control passed as steps instead of seconds
- **File:** `test_mc_pilot.py`
- **Problem:** `"T_control": T / Ts` (= 100) was passed. `reinforce_policy()` internally computes `control_horizon = int(T_control / T_sampling) = int(100 / 0.01) = 10,000` steps — nearly crashed the laptop.
- **Fix:** Changed to `"T_control": T` (= 1.0 seconds). Now `control_horizon = int(1.0/0.01) = 100`.

### Fix 2: opt_steps_list index out of range
- **File:** `test_mc_pilot.py`
- **Problem:** `trial_index` in `reinforce()` starts at `Nexp - 1 = 4`, but lists were sized `[Nopt] * num_trials`. Index 4 out of range for a 3-element list.
- **Fix:** Changed list sizes from `[Nopt] * num_trials` to `[Nopt] * (Nexp + num_trials)`.

### Fix 3: No gradient path to policy parameters
- **File:** `policy_learning/MC_PILCO.py` — `MC_PILOT.apply_policy()`
- **Problem:** `apply_policy()` passed policy speed as GP input. `data_to_gp_input()` strips all inputs, so gradient path from cost → policy params was broken. `RuntimeError: element 0 of tensors does not require grad`.
- **Fix:** Rewrote `apply_policy()` to:
  1. Call policy at t=0 to get scalar speed (with grad)
  2. Convert speed → 3D velocity via differentiable torch ops (atan2, cos, sin)
  3. Embed velocity into initial particle state dims 3:6
  4. GP propagates from that state, preserving gradient chain
- **Result:** `policy weight grad norm = 0.002853 > 0` confirmed.

---

## Performance Improvements

### Change 1: Particle landing freeze (critical)
- **File:** `policy_learning/MC_PILCO.py` — `MC_PILOT.apply_policy()`
- **Problem:** GP propagated particles for all 100 steps regardless of landing. After z≤0, GP kept predicting nonsense physics (ball underground). Cost at T=100 was garbage → cost stuck at ~0.998, no gradients.
- **Fix:** Added `landed` boolean mask. Once a particle's z≤0, its state is frozen (copied from previous step) for all remaining timesteps. Matches paper Section 4.2.2.
- **Result:** Cost dropped from 0.998 → 0.949 at step 0, and fell to 0.882 during optimization. Real learning confirmed.

### Change 2: Widened lc from 0.1 to 0.5m
- **File:** `test_mc_pilot.py`
- **Problem:** With `lc=0.1m`, cost saturates to ~1.0 for any landing error >0.15m. Early training errors are 0.3–1.5m, so gradients were near zero everywhere.
- **Fix:** Increased `lc` from 0.1 to 0.5m. With `lc=0.5`, meaningful gradient exists up to ~1m error.
- **Result:** Cost dropped to ~0.44 in first trial (from ~0.88 plateau with lc=0.1).

### Change 3: Relaxed early stopping
- **File:** `test_mc_pilot.py`
- **Problem:** Early stopping parameters from cartpole were too aggressive for noisy MC cost. Optimizer quit at ~600 steps (out of 1500) due to MC noise triggering the convergence criterion.
- **Fix:** `min_diff_cost` 0.08→0.02, `num_min_diff_cost` 200→400, `min_step` 200→400.
- **Result:** Optimizer now runs ~1200–1500 steps before stopping.

### Change 4: Reduced simulation horizon T from 1.0 to 0.7s
- **File:** `test_mc_pilot.py`
- **Reason:** Ball lands in ~0.58s at max speed (35° elevation, 0.5m release height). With T=1.0s, 42% of GP steps were wasted on frozen particles.
- **Fix:** Reduced T to 0.7s (35 steps at Ts=0.02).
- **Result:** ~30% speedup at no accuracy cost.

### Change 5: Increased timestep Ts from 0.01 to 0.02s
- **File:** `test_mc_pilot.py`
- **Reason:** Sequential Python loop over GP steps was the bottleneck. Halving loop iterations (100→35) gives ~2x speedup. Ball physics are smooth enough for 0.02s.
- **Result:** Time per 100 opt steps: 25s → 8–9s (~3x overall speedup combined with Change 4).

---

## Experiments That Did NOT Work

### Experiment A: Tighten lc to 0.2m (after 10 trials at lc=0.5)
- **Hypothesis:** Policy has converged to ~0.1m errors, tighten lc to sharpen accuracy.
- **Result:** FAILED. Cost jumped back to 0.73–0.81 (was 0.13–0.17). Most particles still land 0.3–0.5m from target in early optimization, which saturates at lc=0.2. Gradient starvation returned.
- **Lesson:** Don't tighten lc until the policy consistently achieves <0.2m errors across the full target range, not just on easy targets.

### Experiment B: Use baseline exploration policy (Eq. 13) instead of random
- **Hypothesis:** Baseline computes analytically correct speed per target (paper Algorithm 1 line 4). Throws near targets → better GP coverage for far-range targets.
- **Result:** FAILED. Hit rate dropped from 50% to 20%. The baseline throws were all near-optimal parabolas with very similar dynamics — the GP trained on 5 nearly-identical trajectories and overfitted to that narrow region. Random exploration provides diverse speeds covering a wider range of velocities and positions, which is better for GP model learning.
- **Lesson:** For the GP to generalize, exploration data needs diversity. Analytically correct ≠ informationally diverse.

### Experiment C: Increase torch threads
- **Hypothesis:** `torch.set_num_threads(1)` leaves CPU idle. Use all cores.
- **Result:** FAILED. Time per 100 steps went from 25s → 35s (40% slower). Thread overhead dominates for small [400×8] tensors. Reverted to 1 thread.

---

## Current Best Configuration

| Parameter | Value | Note |
|-----------|-------|------|
| `lc` | 0.5m | Wider than paper (0.1m) for gradient coverage |
| `Ts` | 0.02s | 2x paper value, ~3x speedup |
| `T` | 0.7s | Trimmed from 1.0s, ball lands by 0.58s |
| `Nopt` | 1500 | Early stopping handles actual termination |
| `min_diff_cost` | 0.02 | Relaxed from 0.08 |
| `num_min_diff_cost` | 400 | Relaxed from 200 |
| `min_step` | 400 | Relaxed from 200 |
| Exploration | Random | Baseline policy gave worse GP coverage |
| Threads | 1 | Multi-threading counterproductive |

## Current Best Scores (seed=1, 10 trials, random exploration, lc=0.5)

| Throw | Phase | Error (m) | Hit (<0.1m) | Target dist |
|-------|-------|-----------|-------------|-------------|
| 1–5 | Explore | 0.28–1.56m | — | random |
| 6 | Trial 1 | 0.61m | miss | 2.10m |
| 7 | Trial 2 | **0.030m** | HIT | 0.82m |
| 8 | Trial 3 | **0.015m** | HIT | 1.41m |
| 9 | Trial 4 | 0.123m | miss | 1.61m |
| 10 | Trial 5 | **0.013m** | HIT | 1.14m |
| 11 | Trial 6 | **0.035m** | HIT | 1.34m |
| 12 | Trial 7 | 0.123m | miss | 1.75m |
| 13 | Trial 8 | 0.732m | miss | 2.15m |
| 14 | Trial 9 | 0.399m | miss | 2.00m |
| 15 | Trial 10 | **0.012m** | HIT | 1.08m |

**Summary:** 5/10 hits (50%). Near/mid targets (<1.8m): mostly hitting. Far targets (>1.9m): mostly missing.

**Root cause of far-target misses:** GP has limited training data at high speeds/long range. Random exploration throws tend to land short, leaving the far-range region under-sampled.

---

### Experiment D: Match paper's center and weight initialization (Eq. 22)
- **Hypothesis:** Paper initializes centers over full Cartesian box `[0, lM] × [-lM·sin(gM), lM·sin(gM)]` and weights over `[-uM, uM]`. Our narrower ranges might be limiting policy coverage.
- **Changes:** Centers: `Px ∈ [0, 2.4]`, `Py ∈ [-1.2, 1.2]` (was `[0.65, 2.4] × [-0.375, 1.2]`). Weights: `[-uM, uM]` (was `[-uM/2, uM/2]`).
- **Result:** FAILED. Hit rate dropped from 50% to 0/10. Cost stuck at ~0.42 across all 10 trials (was reaching 0.13–0.17 before).
- **Root cause:** With lm=0.75m and gM=π/6, targets are always at distance ≥ 0.75m — so `xP < 0.65` is a dead zone. The wider center range wastes ~27% of the 250 centers on regions no target ever occupies, diluting RBF density in the actual target zone. Wider weights may also cause tanh saturation (larger raw values → flatter gradient landscape).
- **Lesson:** The paper's Eq. 22 domain is designed for their robot's geometry (lM=1.75m). Our target domain geometry is different enough that blindly copying it hurts. Centers should cover the actual reachable target arc, not a wider Cartesian box.
