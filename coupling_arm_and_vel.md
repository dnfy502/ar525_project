# Coupled Arm-Ball Physics: Arm Propels the Ball

## Problem Statement

The current `PyBulletThrowingSystem` has **decoupled ballistics**: the arm animates a throw, but at the release moment the ball velocity is **overridden to `v_cmd`** via `p.resetBaseVelocity(ball_id, set_vel=v_cmd)`. The arm motion is purely cosmetic — the ball receives whatever velocity the policy says, regardless of whether the arm could physically deliver it.

This makes the simulation look unrealistic: the throw appears abrupt and mechanical, because there is no physical coupling between the arm's motion and the ball's trajectory.

**Goal:** Make the arm **physically propel the ball** so that the ball's release velocity is determined by the arm's actual end-effector (EE) velocity at the release moment, constrained by joint velocity limits, inertia, and the cubic trajectory planning.

---

## User Review Required

> [!IMPORTANT]  
> **This is a significant architectural change.** The current system trains policies that output a "commanded speed" which is applied directly. In the coupled system, the arm may not be able to achieve the commanded speed exactly, introducing a **sim-to-real gap** between what the policy requests and what the arm delivers. This is actually more realistic and is the exact problem the paper's delay estimation (Modification 2) was designed to address.

> [!WARNING]
> **Training convergence may change.** The coupled system introduces a systematic velocity shortfall (arm tracking is never perfect). The GP will need to learn this as part of the dynamics. The policy will need to learn to overshoot its commanded speed to compensate. This is conceptually identical to the `VelocitySlipNoise` experiment (PB-B) where the policy learned to multiply by `1/(1-α)`, except here the "slip" emerges from real physics rather than being injected.

> [!IMPORTANT]
> **Existing training scripts and results are untouched.** All changes will be in new files or clearly scoped to a new `coupled` mode. The existing `set_vel=v_cmd` behavior remains available as the default.

---

## Open Questions

1. **New subproject folder or in-place?** Should we create `mc-pilot-coupled/` as a new subproject folder (like the wind/elevated/pybullet variants), or add the coupled mode directly inside `mc-pilot-pybullet/`? My recommendation: **in-place inside `mc-pilot-pybullet/`** with a new test script, since all the PyBullet plumbing already exists there. No need to duplicate an entire folder.

2. **Arm trajectory planning quality:** The current cubic polynomial trajectory planner (`plan_throw`) already computes a physically-consistent motion that reaches `qd_release` at the release time. However, PyBullet position control with gains `positionGain=2.0` only achieves ~90% velocity tracking. We have two options:
   - **Option A (Recommended):** Use VELOCITY_CONTROL or TORQUE_CONTROL at the release moment for tighter EE velocity tracking, then let the ball inherit the actual EE velocity
   - **Option B:** Keep POSITION_CONTROL but tune gains aggressively for the throw phase and accept the tracking error as a learnable noise source

3. **Should the `apply_policy` model account for the arm gap?** Currently `apply_policy` uses the 6-D GP to propagate particles. In the coupled system, the GP training data will already contain the arm's tracking error. The GP will implicitly learn `Δv` including the arm shortfall. So no architecture change is needed in `apply_policy` — the GP automatically absorbs the coupling.

---

## Proposed Changes

### Phase 1: Coupled Rollout Mode in PyBulletThrowingSystem

#### [MODIFY] [model_pybullet.py](file:///home/rishang/ar525_project/mc-pilot-pybullet/simulation_class/model_pybullet.py)

Add a `coupled=False` constructor parameter. When `coupled=True`:

1. **No `resetBaseVelocity` at release.** Instead of `arm.release_ball(ball_id, set_vel=v_cmd)`, call `arm.release_ball(ball_id, set_vel=None)` — the ball inherits whatever velocity it has from the physics constraint with the arm.

2. **Improve ball-arm constraint.** Currently the ball is attached via `JOINT_FIXED` and then the constraint is removed at release. The ball's velocity at that moment is whatever PyBullet's constraint solver gives it, which tracks the arm motion.

3. **Record the actual release velocity.** After release, read `p.getBaseVelocity(ball_id)` to get the actual ball velocity (this becomes the GP's first velocity data point instead of `v_cmd`).

The key change in `_simulate_pybullet` (lines 190-203) goes from:
```python
# Current: ball gets v_cmd directly (decoupled)
actual_release_vel = arm.release_ball(ball_id, set_vel=v_cmd)
```
to:
```python
# New coupled mode: ball gets whatever the arm gives it
arm.release_ball(ball_id, set_vel=None)
actual_release_vel = p.getBaseVelocity(ball_id)[0]  # actual physics
```

---

#### [MODIFY] [arm_controller.py](file:///home/rishang/ar525_project/mc-pilot-pybullet/robot_arm/arm_controller.py)

Tune the throw execution for tighter velocity tracking in coupled mode:

1. **Add velocity-feed-forward in `step()`:** The current `POSITION_CONTROL` fights to reach position targets and uses `targetVelocities` only as feed-forward hints. For the throw phase (windup → release), switch to `VELOCITY_CONTROL` mode in the final few steps before release, where tracking the correct joint *velocity* matters more than position.

2. **Increase max forces during throw phase:** Multiply `maxForce` during the throw phase to let the arm accelerate harder into the release.

3. **Add `step_throw()` method** that uses higher gains or velocity-priority control during the critical throw phase, to maximize EE velocity fidelity at the release moment.

---

### Phase 2: New Test Script for Coupled Training

#### [NEW] [test_mc_pilot_pb_coupled.py](file:///home/rishang/ar525_project/mc-pilot-pybullet/test_mc_pilot_pb_coupled.py)

New training script modeled on `test_mc_pilot_pb_A.py` but using `coupled=True`. Key differences:

- `PyBulletThrowingSystem(coupled=True, robot_name="kuka_iiwa")`
- `uM` may need to be reduced to account for the EE velocity ceiling (arm can't deliver 3.5 m/s — maybe 2.0-2.5 m/s achievable)
- Target range `lm/lM` calibrated to the achievable range with coupled physics
- Exploration strategy adjusted (stratified over the achievable speed range)

This script runs the full MC-PILOT training loop identically to the existing scripts, but the GP learns from coupled physics data instead of `v_cmd`-overridden data.

---

### Phase 3: Updated Demo Visualization

#### [MODIFY] [demo_pybullet_gui.py](file:///home/rishang/ar525_project/mc-pilot-pybullet/demo_pybullet_gui.py)

Add `--coupled` flag. When set:
- Uses `arm.release_ball(ball_id, set_vel=None)` instead of `set_vel=v_cmd`
- Ball velocity at release is printed (showing the actual vs commanded gap)
- Throw looks physically realistic: the arm accelerates, the ball rides the EE, and at release it flies with the momentum the arm gave it

---

### Phase 4: Speed Calibration Script

#### [NEW] [calibrate_coupled_arm.py](file:///home/rishang/ar525_project/mc-pilot-pybullet/calibrate_coupled_arm.py)

Standalone script that:
1. Plans throws at various `v_cmd` values (0.5 to 3.5 m/s)
2. For each, runs the arm trajectory in coupled mode and measures actual EE velocity at release
3. Prints a table: `v_cmd` → `v_actual` → ratio → landing range
4. This tells us: what `uM` is achievable, what `lm/lM` to use, and what systematic "slip" the arm introduces

This is critical for setting hyperparameters correctly for the coupled training script.

---

## Implementation Sequence

```
1. calibrate_coupled_arm.py  — measure what the arm can actually deliver
2. model_pybullet.py         — add coupled=True mode
3. arm_controller.py         — tune throw execution for velocity tracking
4. test_mc_pilot_pb_coupled.py — training script with coupled physics
5. demo_pybullet_gui.py      — visual demo in coupled mode
6. Run calibration → set hyperparameters → run training → run demo
```

---

## Verification Plan

### Automated Tests
1. **Calibration table:** Run `calibrate_coupled_arm.py` to verify the arm can deliver meaningful EE velocities (>1 m/s) at the release moment
2. **Parity check:** `test_mc_pilot_pb_coupled.py` with `coupled=False` should reproduce `test_mc_pilot_pb_A.py` results exactly (zero regression)
3. **Coupled training:** Run `test_mc_pilot_pb_coupled.py` with `coupled=True`, seed=1, 5 trials — verify cost decreases and landing errors < 15cm

### Manual Verification
1. **Visual demo:** Run `demo_pybullet_gui.py --coupled` and visually confirm the ball is propelled by the arm's motion (no abrupt velocity jump)
2. **Record video:** Use `--record demo_coupled.mp4` to capture the smooth coupled throw for comparison with the decoupled version
