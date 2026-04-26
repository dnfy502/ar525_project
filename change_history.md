# MC-PILOT Change History
AR525 Group-3, IIT Mandi

---

## Baseline Bugs Fixed

### Documentation Update: Mentor meeting crash brief (LaTeX)
- **Files:** `mentor_meeting_brief.tex`, `change_history.md`
- **Problem:** A complete but concise handover was needed for mentor discussion so a teammate with no prior participation can answer technical and implementation-level questions confidently.
- **Change:** Added a compact LaTeX briefing that covers architecture, code map, algorithm flow, key equations, critical failure modes and fixes, result summary across configs, scope limitations, rapid Q&A, and runnable commands, including real code snippets and pseudocode.
- **Result:** The repository now has a ready-to-compile mentor prep document with high-density technical coverage for defense-style questioning.

### Documentation Update: Repository-verified Graphify map and file crosswalk
- **Files:** `graphify-out/graph.html`, `graphify-out/graph.json`, `graphify-out/GRAPH_REPORT.md`, `change_history.md`
- **Problem:** The previous Graphify architecture view reflected an idealized four-folder layout from the report, but this checkout currently contains `mc-pilot/` and `mc-pilot-elevated/` as primary project folders, with pybullet-related scripts located inside `mc-pilot/`. This made it hard to trace report claims to concrete files.
- **Change:** Rebuilt Graphify outputs to be repo-verified, explicitly marking which sub-project folders/files are present vs report-mentioned-only in this branch, and added a detailed file-level crosswalk for sub-project mapping, failure-mode fix locations, shared core modules, stratified exploration implementation, and wind-state augmentation status.
- **Result:** The Graphify artifacts now provide an accurate, auditable bridge from report architecture to real code locations in the current repository snapshot.

### Documentation Update: Graphify layered architecture diagram refreshed
- **Files:** `graphify-out/graph.html`, `graphify-out/graph.json`, `graphify-out/GRAPH_REPORT.md`, `change_history.md`
- **Problem:** The previous Graphify visualization mixed low-level extracted entities and did not clearly present the exact four-subproject architecture, inheritance arrows, shared core files, and wind cross-cutting scope requested for repository communication.
- **Change:** Replaced Graphify outputs with a curated layered architecture view showing: `mc-pilot/` as the base, `mc-pilot-elevated/` and `mc-pilot-pybullet/` as layer-2 extensions, `mc-pilot-pb-elevated/` as the layer-3 combined branch, plus explicit `shared_across` and wind `modifies` links to all four.
- **Result:** The Graphify artifacts now communicate the repository structure and dependency relationships directly and unambiguously.

### Documentation Update: Added report defense checklist
- **Files:** `report_defense_checklist.md`, `change_history.md`
- **Problem:** The project had detailed implementation notes and experiment logs, but no defense-focused document explaining how to present the work strongly while staying accurate about scope, contributions, and limitations.
- **Change:** Added a dedicated checklist covering what can be claimed confidently, what should not be overclaimed, how to present the release-height extension truthfully, likely defense questions, and the main limitations to admit early.
- **Result:** The repo now includes a ready-to-use defense guide that helps present the project as impressive, rigorous, and honest.

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

---

### Experiment G: Revert lc from 0.5 m to paper's 0.1 m
- **File:** `test_mc_pilot.py`
- **Hypothesis:** lc was widened to 0.5 because early errors (0.3–1.5m) saturated the cost at lc=0.1, killing gradients. That saturation was caused by physically unreachable targets (lM=2.4). With lM=1.75 (all targets reachable) and a well-trained GP, exploration throws should land much closer to targets — keeping cost below saturation at lc=0.1. Sharper cost function should enable finer accuracy and lower final cost.
- **Changes:** `lc = 0.5 → 0.1`
- **Result:** FAILED. 0/10 hits. Cost floor stuck at ~0.80–0.81 across all 10 trials (never below 0.79). Errors actually worsened after Trial 5 (0.7–0.86m), suggesting policy wandering rather than learning.
- **Root cause:** Exploration throws still land 0.27–1.09m from targets (random speed, GP untrained at that point). At lc=0.1, an error of 0.27m gives cost ≈ 0.999 — essentially saturated. Nearly all particles are at cost~1.0 before Trial 1 even begins, so gradients are near zero and the optimizer can only scratch down to ~0.80 before signal disappears. The GP needs several trials at lc=0.5 to become accurate enough that lc=0.1 is viable.
- **Lesson:** lc=0.1 requires the policy to already be near-accurate before it provides useful gradients. It cannot bootstrap from scratch with random exploration. lc=0.5 is necessary for initial learning.

---

### Experiment F: Revert RBF center and weight init to paper's Eq. 22 (with corrected lM=1.75)
- **File:** `test_mc_pilot.py`
- **Hypothesis:** Experiment D tried this with lM=2.4 and failed (0/10 hits) — dead zone (Px<0.65) wasted 27% of centers on unreachable targets, and impossible far targets created noisy gradients. With lM=1.75 (all targets reachable), the gradient landscape is clean. Wider init may improve boundary smoothness and prevent local minima.
- **Changes:**
  - Centers: `Px ∈ [lm·cos(gM), lM] → [0, lM]`, `Py ∈ [lm·sin(-gM), lM·sin(gM)] → [-lM·sin(gM), lM·sin(gM)]`
  - Weights: `[-uM/2, uM/2] → [-uM, uM]`
- **Note:** Dead zone (Px < ~0.65m) is still ~37% of Px range, but all sampled targets are now within physics reach so gradients are meaningful.
- **Result:** FAILED. 2/10 hits. Cost stuck at ~0.45–0.50 across all 10 trials (Trial 1: 0.73→0.49, Trials 2–10: plateau immediately at 0.45–0.50, never escaping). Same failure mode as Experiment D despite correct lM=1.75.
- **Root cause:** The target domain is strictly a sector between 0.75m and 1.75m. By starting centers at Px=0, ~37% of the 250 basis functions land in a dead zone where targets never spawn. These become "ghost centers" — never trained by any real throw, they output persistent random background noise that corrupts the RBF sum across the entire target arc. This leaves only ~160 centers to cover the actual target region, which isn't enough density to sculpt a smooth, accurate speed ramp. The optimizer gets stuck at cost ~0.44–0.53 because the policy landscape is too noisy and under-resolved to optimize through.
- **Sub-experiment F1 additionally had:** wide weights [-uM, uM] causing tanh saturation on top of the center problem.
- **Conclusion:** Paper's Eq. 22 center domain is designed for their geometry and cannot be ported directly. Centers must begin at the actual target boundary (~lm·cos(gM)≈0.65m), ensuring all 250 basis functions cover regions where targets actually appear.

---

### Experiment E: Reduce lM from 2.4 m to 1.75 m (paper's value)
- **File:** `test_mc_pilot.py`
- **Hypothesis:** Paper defines target domain as l ∈ [0.75, 1.75] m. At lM=2.4 m with uM=3.5 m/s and α=35°, far targets may be physically unreachable — policy penalized for impossible throws. Reverting to paper's lM should eliminate far-target misses.
- **Changes:** `lM = 2.4 → 1.75`. RBF centers upper bound auto-updates (uses `lM` variable).
- **Result:** SUCCESS. 5/5 hits (100%). Errors: 85mm, 33mm, 23mm, 17mm, 8mm. Cost converged from 0.74 → 0.009 in Trial 1, reaching 0.001 by Trial 5. Trial 5 hit early stopping at step 1204 — policy fully converged. Far-target misses eliminated entirely.

---

## Exploration 1: Variable Platform Height (Elevated Targets)

### Setup
- **Folder:** `mc-pilot-elevated/` — full copy of `mc-pilot/`, original untouched
- **Idea:** Robot platform is at different heights above ground. Only `RELEASE_POS[2]`, `lM`, and `T` change per config. All other parameters (policy, GP, cost, lm, gM, M, Nb, lc) are identical to Config A.
- **RBF centers auto-update** — code already uses `lM` as a variable.
- **Max ranges verified** by running `check_max_range.py` with ThrowingSystem at uM=3.5, α=35°.

| Config | z_release | Max range (with drag) | lM used | T |
|--------|-----------|----------------------|---------|---|
| A | 0.5m | 1.647m | 1.75m (validated) | 0.70s |
| B | 1.0m | 1.973m | 1.90m | 0.85s |
| C | 1.5m | 2.233m | 2.15m | 0.95s |
| D | 2.0m | 2.457m | 2.35m | 1.00s |
| E | 0.0m | 1.156m | 1.10m | 0.55s |

**Scripts:** `test_mc_pilot_b.py` through `test_mc_pilot_e.py`
**Results dirs:** `results_mc_pilot_{b,c,d,e}/{seed}/`

### Config B, seed=1 (random exploration): FAILED — 0/10 hits
- All trial throws landed at x≈1.90m (max range) regardless of target
- Cost stuck at 0.70–0.73 across all 10 trials, never dropping
- Root cause: seed=1 exploration drew very low speeds (throws landing at 0.245m, 0.408m). GP never saw high-speed dynamics. Policy found max-speed local minimum.

### Fix: Stratified Exploration — `Stratified_Throwing_Exploration`
- **File:** `policy_learning/Policy.py` — added `Stratified_Throwing_Exploration` class
- Divides [0, uM] into Nexp=5 equal bands [0–0.7, 0.7–1.4, 1.4–2.1, 2.1–2.8, 2.8–3.5 m/s]. Each exploration throw samples from the next band in order. GP guaranteed to see the full speed range regardless of seed.
- **Adopted as default for all elevated configs (B–E).**

### Config B, seed=2 (random exploration): SUCCESS — 10/10 hits
- Got lucky: exploration throw 4 landed 18mm from target, giving GP good near-range data
- Trial 1: cost 0.882 → 0.005. Fully converged by Trial 3 (cost <0.001)
- Mean error: 20mm, min: 16mm
- Confirms z=1.0m is learnable when GP has good speed coverage

### Config B-Strat, seed=1 (stratified exploration): SUCCESS — 10/10 hits
- Same seed that failed completely with random exploration
- Trial 1: cost 0.811 → 0.003. Fully converged by Trial 3
- Mean error: 20mm, min: 6mm
- **Proves stratified exploration is seed-independent and robust**
- Lesson: random exploration is fragile at elevated heights because the speed-to-range mapping is less forgiving. Stratified is the reliable default.

### Config C-Strat, seed=1 (stratified, z=1.5m): SUCCESS — 10/10 hits
- Exploration errors: 0.17–1.56m (good spread across all speeds)
- Trial 1: cost 0.809 → 0.085. Fully converged by Trial 2 (cost 0.0005 by Trial 3)
- Errors (Trials 1–10): 16, 5, 12, 22, 27, 7, 11, 11, 24, 8 mm. Min: 5mm, max: 27mm.
- Target distances ranged from 0.75m to 2.06m — policy generalises across full arc
- Clean convergence: cost drops from 0.08 (T2) to 0.0005 by T3 and stays there

### Config D-Strat, seed=1 (stratified, z=2.0m): SUCCESS — 10/10 hits
- Exploration errors: 0.15–0.86m
- Trial 1: cost 0.827 → 0.024 (faster first-trial drop than C; better GP from z=2.0m longer flight)
- Errors (Trials 1–10): 7, 19, 15, 25, 21, 12, 18, 9, 29, 9 mm. Min: 7mm, max: 29mm.
- Target distances ranged from 0.82m to 2.20m — covers nearly the full lM=2.35m range
- Best first-trial convergence of all elevated configs (0.827 → 0.024 in one pass)

### Experiment H: Config E-Strat, seed=1 (stratified [0, uM], z=0.0m): FAILED — 0/10 hits
- Cost stuck at ~0.25 throughout all 10 trials (Trial 1: 0.663→0.172; Trials 2–10: plateaus immediately at 0.23–0.28, barely moving)
- Errors: 105–396mm, all well above 100mm threshold. Policy not converging at all.
- **Root cause: speed-range compression at z=0.0m.** From z=0.0 at α=35°, targets [0.75, 1.10]m require release speeds [~2.80, ~3.39] m/s (only 17% of [0, 3.5] m/s). With standard stratified exploration (5 equal bands of 0.7 m/s), throws from bands 1–4 land at x<0.75m and fall outside the target zone entirely. Only band 5 ([2.8, 3.5]) produces throws that reach the target. GP trained on 4 irrelevant short-range throws and only 1 useful throw — far too little data in the actual target dynamics regime.
- Failure mode differs from B-Random: B stuck at cost~0.70 (max-speed local min). E stuck at cost~0.25 (optimizer flailing in flat gradient landscape, no clear direction). Cost init and final are nearly identical by Trial 3 onward — gradient signal is effectively zero.
- **Lesson:** Stratified exploration over full [0, uM] only works when the target range spans a substantial fraction of the speed range. When targets cluster near max range (as in E), a targeted exploration strategy is needed.

### Fix attempt 1 for E: Config E-Strat2 — targeted stratification over [2.5, uM]
- **Files:** `policy_learning/Policy.py` (added `u_min` param to `Stratified_Throwing_Exploration`), `test_mc_pilot_e_strat2.py`
- **Change:** Exploration bands now span [2.5, 3.5] m/s. All 5 throws land in target zone (errors 16–325mm vs 350–924mm before). 2 exploration near-hits.
- **Result: FAILED — same cost plateau at ~0.25.** Better exploration didn't help.
- **Diagnosis:** Exploration quality was NOT the primary bottleneck. Root cause is GP data scarcity: T=0.55s gives only 137 raw training points before Trial 1 (vs 250 for D-Strat). GP simulation consistently predicts ~0.38m errors while real throws achieve ~0.25m — inaccurate GP means optimizer stalls regardless of seed.

### Fix attempt 2 for E: Config E-Strat3 — Nexp=10, targeted stratification: FAILED
- **File:** `test_mc_pilot_e_strat3.py`
- **Change:** `Nexp=5 → 10`. 275 training points, finer speed bands [2.5–3.5].
- **Result:** 0/10 hits. Same cost plateau at ~0.25. Policy outputs exactly 3.500 m/s for EVERY target regardless of distance. **Confirmed: tanh saturation at max speed.**
- **Root cause identified:** With lc=0.5m > target range (0.35m), max-speed cost ≈ 0.25 (ball overshoots lM by ~0.3m). The cost landscape is so flat that max-speed is a stable local minimum — optimizer gradient can't distinguish between "slightly wrong speed" and "correct speed" within the target range. GP data quantity was not the issue.

### Fix attempt 3 for E: Config E-Strat4 — lc=0.25, Nexp=10, targeted stratification
- **File:** `test_mc_pilot_e_strat4.py`
- **Change:** `lc=0.5 → 0.25`. With lc=0.25, max-speed cost rises to ~0.69 (clearly penalised), near-optimal cost is ~0.05. Steep gradient forces optimizer away from saturation. Nexp=10 provides enough GP data (275 pts) to avoid gradient starvation at lc=0.25.
- **Result:** FAILED — 0/10 hits. Ball lands at exactly **1.156m** for every single trial regardless of target (targets ranged 0.80–1.02m). Cost stuck at 0.60–0.64 with no monotone trend (trial 3 cost actually increased). lc=0.25 was not the issue.
- **Root cause (revised):** lc addressed the wrong symptom. The actual failure is **RBF policy blindness**: with `lengthscales_init=[1.0, 1.0]`, the RBF sensitivity between Config E's target extremes (0.75m vs 1.10m, distance=0.35m) is `exp(-0.5*(0.35/1.0)²) = 0.941` — all targets produce nearly identical activation vectors `φ(P)`. The policy output `speed = squash(W·φ(P))` is therefore constant regardless of target. It converges to max speed because Gaussian MC particles with targets beyond lM=1.10m bias the "optimal constant" toward max speed. The lengthscale gradient is also near-zero (same reason), so the optimizer cannot self-correct. lc, Nexp, exploration strategy all addressed symptoms; this blindness was the root cause throughout E-Strat1–4.
- **Comparison:** Config A has a 1.0m target range with ls=1.0m → sensitivity=0.607 (targets distinguishable). Config E has 0.35m range → 0.941 (targets indistinguishable). Same lengthscale, very different resolution.

### Fix attempt 4 for E: Config E-Strat5 — lengthscales_init=[0.15, 0.15]
- **File:** `test_mc_pilot_e_strat5.py`
- **Change:** `lengthscales_init=[1.0, 1.0] → [0.15, 0.15]`. At ls=0.15m, sensitivity between Config E's extremes = `exp(-0.5*(0.35/0.15)²) = 0.066` — targets are clearly distinguishable, same resolution as Config A. Policy can now map near targets → low speed, far targets → high speed. All other settings from E-Strat4 retained (lc=0.25, Nexp=10, strat [2.5, 3.5]).
- **Result:** SUCCESS — **10/10 hits**. Ball distance per trial now tracks target distance (0.80–1.04m), confirming policy correctly varies speed by target — complete contrast to E-Strat4 where every throw was 1.156m. Trial 1: cost 0.786→0.010 (strong first-trial convergence). Trial 2 onward: cost already starts at 0.013–0.020, fully converged by trial 2. Mean error: 26mm, min: 20mm, max: 33mm. Tightest error band of all elevated configs.
- **Lesson:** For narrow target ranges, `lengthscales_init` must be scaled to the target range, not left at the paper's default of [1.0, 1.0]. Rule: `ls ≈ 0.1–0.2 × target_range` gives comparable resolution to configs that work at ls=1.0m with 1.0m range.

### General finding: lengthscales must scale with target range

The paper uses `lengthscales_init=[1.0, 1.0]` throughout. This works when the target range is ~1m wide but silently breaks for narrow target domains. The RBF sensitivity between two target positions at distance `d` apart is `exp(-0.5*(d/ls)²)`. When this is close to 1.0, the policy cannot distinguish between those targets and collapses to a constant output.

**Rule of thumb:** `ls ≈ 0.15 × target_range` keeps sensitivity at ~0.066 between the domain extremes — comparable to what Config A achieves with ls=1.0m over a 1.0m range. For any config with a narrow target arc, scale down `lengthscales_init` proportionally before running.

| Config | Target range | Recommended ls | Sensitivity at extremes |
|--------|-------------|----------------|------------------------|
| A–D (range ≥ 1.0m) | ≥ 1.0m | 1.0m (paper default) | ≤ 0.607 ✓ |
| E (range 0.35m) | 0.35m | **0.15m** | 0.066 ✓ |

This is the only parameter the paper does not adapt to geometry. Everything else (lM, T, exploration strategy) does adapt.

---

### Elevated configs summary

| Config | z_release | lM | Exploration | Seed | Result |
|--------|-----------|-----|-------------|------|--------|
| A | 0.5m | 1.75m | Random | 1 | 5/5 hits, mean err ~33mm |
| B | 1.0m | 1.90m | Random | 1 | 0/10 FAILED (bad seed) |
| B | 1.0m | 1.90m | Random | 2 | 10/10, mean err 20mm |
| B-Strat | 1.0m | 1.90m | Stratified | 1 | 10/10, mean err 20mm |
| C-Strat | 1.5m | 2.15m | Stratified | 1 | **10/10**, mean err ~14mm |
| D-Strat | 2.0m | 2.35m | Stratified | 1 | **10/10**, mean err ~16mm |
| E-Strat | 0.0m | 1.10m | Stratified [0, 3.5] | 1 | 0/10 FAILED — speed-range compression |
| E-Strat2 | 0.0m | 1.10m | Stratified [2.5, 3.5], Nexp=5 | 1 | 0/10 FAILED — GP data-starved |
| E-Strat3 | 0.0m | 1.10m | Stratified [2.5, 3.5], Nexp=10 | 1 | 0/10 FAILED — tanh saturation at max speed |
| E-Strat4 | 0.0m | 1.10m | Strat [2.5,3.5], Nexp=10, lc=0.25 | 1 | 0/10 FAILED — RBF blindness (ls=1m >> target range 0.35m) |
| E-Strat5 | 0.0m | 1.10m | Strat [2.5,3.5], Nexp=10, lc=0.25, ls=0.15 | 1 | **10/10**, mean err 26mm |

---

## Current Best Configuration

| Parameter | Value | Note |
|-----------|-------|------|
| `lc` | 0.5m | Wider than paper (0.1m) for gradient coverage |
| `Ts` | 0.02s | 2x paper value, ~3x speedup |
| `T` | 0.7s | Trimmed from 1.0s, ball lands by 0.58s |
| `lM` | 1.75m | Paper's value; 2.4m was beyond physical reach |
| `Nopt` | 1500 | Early stopping handles actual termination |
| `min_diff_cost` | 0.02 | Relaxed from 0.08 |
| `num_min_diff_cost` | 400 | Relaxed from 200 |
| `min_step` | 400 | Relaxed from 200 |
| Exploration | Random | Baseline policy gave worse GP coverage |
| Threads | 1 | Multi-threading counterproductive |

## Current Best Scores (seed=1, 5 trials, random exploration, lc=0.5, lM=1.75)

| Throw | Phase | Error (m) | Hit (<0.1m) | Target dist |
|-------|-------|-----------|-------------|-------------|
| 1–5 | Explore | 0.02–1.09m | — (1 accidental hit) | random |
| 6 | Trial 1 | **0.085m** | HIT | 1.66m |
| 7 | Trial 2 | **0.033m** | HIT | 1.60m |
| 8 | Trial 3 | **0.023m** | HIT | 1.18m |
| 9 | Trial 4 | **0.017m** | HIT | 0.79m |
| 10 | Trial 5 | **0.008m** | HIT | 1.10m |

**Summary:** 5/5 hits (100%). Cost converged from 0.74 → 0.009 (Trial 1) to 0.001 (Trial 5). Policy fully converged by Trial 5 (early stopped at step 1204).

---

### Experiment D: Match paper's center and weight initialization (Eq. 22)
- **Hypothesis:** Paper initializes centers over full Cartesian box `[0, lM] × [-lM·sin(gM), lM·sin(gM)]` and weights over `[-uM, uM]`. Our narrower ranges might be limiting policy coverage.
- **Changes:** Centers: `Px ∈ [0, 2.4]`, `Py ∈ [-1.2, 1.2]` (was `[0.65, 2.4] × [-0.375, 1.2]`). Weights: `[-uM, uM]` (was `[-uM/2, uM/2]`).
- **Result:** FAILED. Hit rate dropped from 50% to 0/10. Cost stuck at ~0.42 across all 10 trials (was reaching 0.13–0.17 before).
- **Root cause:** With lm=0.75m and gM=π/6, targets are always at distance ≥ 0.75m — so `xP < 0.65` is a dead zone. The wider center range wastes ~27% of the 250 centers on regions no target ever occupies, diluting RBF density in the actual target zone. Wider weights may also cause tanh saturation (larger raw values → flatter gradient landscape).
- **Lesson:** The paper's Eq. 22 domain is designed for their robot's geometry (lM=1.75m). Our target domain geometry is different enough that blindly copying it hurts. Centers should cover the actual reachable target arc, not a wider Cartesian box.
