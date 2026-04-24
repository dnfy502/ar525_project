# MC-PILOT Implementation Plan

## Context

We have MC-PILCO working (cartpole validated). Now we adapt it into MC-PILOT — a single-shot throwing policy that maps target position to release velocity. The GP learns free-flight ball dynamics, the policy fires once at t=0, and cost is evaluated at landing.

**User decisions:**
* Start with numpy ballistic simulator, add MuJoCo/PyBullet later for arm visualization
* Policy outputs 1D speed (direction derived from target geometry)
* Copy MC-PILCO into a new mc-pilot/ folder and modify there; original stays as reference

---

## File Structure

```text
project/
  MC-PILCO/                  # UNTOUCHED — reference copy
  mc-pilot/                  # NEW — full copy of MC-PILCO, then modified
    policy_learning/
      MC_PILCO.py            # MODIFY — override apply_policy, get_data_from_system
      Policy.py              # MODIFY — add Throwing_Policy, Random_Throwing_Exploration
      Cost_function.py       # MODIFY — add Throwing_Cost
    model_learning/
      Model_learning.py      # MODIFY — add Ballistic_Model_learning
    simulation_class/
      model.py               # MODIFY — add ThrowingSystem class
      ode_systems.py         # NO CHANGE (not using ODE for throwing)
    gpr_lib/                 # NO CHANGE
    test_mc_pilot.py         # NEW — main test script (mirrors test_mcpilco_cartpole.py)
```

---

## Components (6 pieces)

### 1. ThrowingSystem (simulation_class/model.py)

Replaces the ODE-based Model for data collection. Simulates ball free-flight using Eq. 35 from the paper.

* **Input:** augmented state [x,y,z, vx,vy,vz, Px,Py], policy, T, dt, noise
* **Behavior:** calls policy ONCE at t=0 to get release speed → converts to 3D velocity using target geometry → simulates ballistic flight with drag → detects landing (z ≤ 0)
* **Output:** (noisy_states, inputs, clean_states) — same signature as Model.rollout()
* **Physics (Eq. 35):** $F_{drag} = -0.5 \cdot \rho \cdot C_D(Re) \cdot A \cdot ||v|| \cdot v$, integrated with RK4 or Euler
* **Wind (Exploration 2, later):** add $F_{wind}$ term, configurable per-instance

**Geometry for 1D speed → 3D velocity:**
* Target is at (Px, Py, 0) relative to release point
* Horizontal distance $d = \sqrt{Px^2 + Py^2}$, azimuth $\phi = \operatorname{atan2}(Py, Px)$
* Elevation angle $\alpha$ derived from ballistic trajectory equation (fixed or optimized later)
* $vx = v \cdot \cos(\alpha) \cdot \cos(\phi)$, $vy = v \cdot \cos(\alpha) \cdot \sin(\phi)$, $vz = v \cdot \sin(\alpha)$

### 2. Ballistic_Model_learning (model_learning/Model_learning.py)

New subclass added to Model\_learning.py. GP learns velocity changes during free flight.

* **num_gp** = 3 (one each for $\Delta vx, \Delta vy, \Delta vz$)
* **GP input** = [x,y,z, vx,vy,vz] (6D — ball state only, strips target P and zero-inputs)
* **GP output** = velocity deltas between consecutive timesteps
* **State reconstruction:** $v_{t+1} = v_t + \Delta v$, $p_{t+1} = p_t + Ts \cdot v_t + (Ts/2) \cdot \Delta v$
* **Augmented state handling:** target dims (6:8) are copied forward unchanged in `get_next_state_from_gp_output()`
* Subclasses `Speed_Model_learning_RBF` pattern with `vel_indeces=[3,4,5]`, `not_vel_indeces=[0,1,2]`, no angles

### 3. Throwing_Policy (policy_learning/Policy.py)

New class added to Policy.py. Single-shot RBF policy: $\pi(P) \to \text{speed}$.

* Subclasses `Sum_of_gaussians` with `state_dim=2` (target Px,Py), `input_dim=1` (speed)
* **forward(states, t, p_dropout):**
    * If t > 0: return zeros (no control during flight)
    * If t == 0: extract target P from augmented state (last 2 dims), pass to RBF network
* **Squashing:** $(uM/2) \cdot (\tanh(raw) + 1) \to$ output in $[0, uM]$ (paper's Eq. 21)
* **num_basis** = 250, centers initialized randomly in target range $[\ell m, \ell M] \times [-\gamma M, \gamma M]$

Also add **Random_Throwing_Exploration**: returns random speed $\in [0, uM]$ at t=0, zeros after.

### 4. Throwing_Cost (policy_learning/Cost_function.py)

New cost function added to Cost\_function.py. Evaluates landing accuracy.

* **Cost (Eq. 16):** $c = 1 - \exp(-||p_{landing} - P||^2 / \ell c^2)$
* Only evaluated at final timestep (or when ball lands, whichever is detected)
* **position_indices** = [0,1] (ball x,y), **target_indices** = [6,7] (Px,Py)
* For ground-plane baseline: z-error ignored ($\Sigma c = \operatorname{diag}(1/\ell c, 1/\ell c, 0)$)
* Returns cost tensor [T, M, 1] with zeros for t < T, real cost at t = T

### 5. MC_PILOT overrides (policy_learning/MC_PILCO.py)

Modify the existing MC\_PILCO class (or add MC\_PILOT subclass) with throwing-specific logic.

**Key overrides:**

a) **get_data_from_system()** — sample random target P, build augmented initial state [release\_pos, 0,0,0, Px, Py], call `ThrowingSystem.rollout()`

b) **apply_policy() particle initialization** — each of M=400 particles gets a different random target P sampled from $U([\ell m, \ell M] \times [-\gamma M, \gamma M])$. Ball position initialized at release point with small noise. Target dims set per-particle.

c) **reinforce() loop** — adapt initial state handling: instead of fixed Gaussian around [0,0,0,0], use augmented state with random targets per episode.

### 6. Test script (test_mc_pilot.py)

Mirrors `test_mcpilco_cartpole.py` structure. Configures all components with Table 1 parameters:

* Nexp=5, Nopt=1500, M=400, Nb=250
* uM=3.5, Ts=0.01, T=1.0, $\ell c$=0.1
* Target range: [0.75, 2.4]m distance, $[-\pi/6, \pi/6]$ lateral angle
* Release position: fixed (e.g., [0, 0, 0.5] — 0.5m above ground)
* Ball parameters: mass=0.0577kg (tennis ball), radius=0.0327m, $C_D \approx 0.47$

---

## Data Flow

For each trial:
1.  **EXPLORE (5 throws):**
    * Sample random target P → build augmented s0
    * → `ThrowingSystem.rollout(s0, random_policy)` → ball trajectory
    * → `Ballistic_Model_learning.add_data(trajectory)` [strips target, stores ball-only pairs]
2.  **TRAIN GP:**
    * → `Ballistic_Model_learning.reinforce_model()` [optimize marginal likelihood]
3.  **OPTIMIZE POLICY (1500 steps):**
    * For each step:
        * → Sample 400 particles with diverse targets
        * → t=0: `Throwing_Policy(augmented_state)` → release speed
        * → t=1..100: GP propagates ball state, policy returns zeros
        * → `Throwing_Cost` at t=T: landing error
        * → Backprop through GP chain → update policy weights
4.  **TEST:**
    * → Sample new target → `ThrowingSystem.rollout(s0, trained_policy)`
    * → Measure landing error

---

## Implementation Order

| Step | What | Files touched |
| :--- | :--- | :--- |
| 0 | Copy MC-PILCO/ → mc-pilot/ | shell |
| 1 | ThrowingSystem (numpy ballistics with drag) | simulation\_class/model.py |
| 2 | Ballistic\_Model\_learning (GP for ball dynamics) | model\_learning/Model\_learning.py |
| 3 | Throwing\_Policy + Random\_Throwing\_Exploration | policy\_learning/Policy.py |
| 4 | Throwing\_Cost | policy\_learning/Cost\_function.py |
| 5 | MC\_PILOT overrides (augmented state, target sampling) | policy\_learning/MC\_PILCO.py |
| 6 | test\_mc\_pilot.py (wire everything, run baseline) | test\_mc\_pilot.py (new) |
| 7 | Validate: reproduce ~100% accuracy on ground targets | — |
| 8 | Exploration 1: elevated targets (expand policy to 3D, modify cost) | Policy.py, Cost\_function.py |
| 9 | Exploration 2: wind (add wind to ThrowingSystem, test GP adaptation) | model.py |
| 10 | Visualization: MuJoCo/PyBullet arm demo | new files |

---

## Verification

1.  **Unit test ThrowingSystem:** throw straight at known target with no drag → verify exact parabolic landing
2.  **Unit test with drag:** compare against analytical drag solution for known CD
3.  **GP sanity check:** after 5 exploration throws, GP MSE on training data should be < 0.001
4.  **Policy convergence:** cost should decrease monotonically over 1500 optimization steps
5.  **End-to-end:** after 2-3 trials (10-15 throws), accuracy on random targets should approach >80%
6.  **Paper comparison:** Table 1 conditions → near-100% accuracy within ~10 throws

