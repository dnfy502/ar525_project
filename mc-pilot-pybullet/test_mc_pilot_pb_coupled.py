"""
MC-PILOT training with coupled arm-ball physics.

The arm physically propels the ball — no resetBaseVelocity override.
Ball velocity at release equals the arm's actual EE velocity, constrained
by joint velocity limits, inertia, and the cubic trajectory planner.

Calibrated for kuka_iiwa:
  - Max achievable EE speed: ~0.89 m/s (joint velocity limited)
  - Usable range: 0.50–0.77 m from release position
  - uM=1.0 m/s (arm clips anything higher to ~0.89 m/s)

Usage:
  python test_mc_pilot_pb_coupled.py -seed 1 -num_trials 5
  python test_mc_pilot_pb_coupled.py -seed 1 -num_trials 10 -coupled 1
  python test_mc_pilot_pb_coupled.py -seed 1 -num_trials 5 -coupled 0   # decoupled baseline
"""

import argparse
import os
import pickle as pkl
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import model_learning.Model_learning as gp_model
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO as MC_PILCO
import policy_learning.Policy as Policy
from simulation_class.model_pybullet import PyBulletThrowingSystem

parser = argparse.ArgumentParser()
parser.add_argument("-seed", type=int, default=1)
parser.add_argument("-num_trials", type=int, default=5)
parser.add_argument("-coupled", type=int, default=1,
                    help="1=coupled (arm propels ball), 0=decoupled (set_vel=v_cmd)")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

coupled = bool(args.coupled)

# --- Coupled hyperparameters (calibrated from calibrate_coupled_arm.py) ---
uM = 1.0          # Max commandable speed (arm clips at ~0.89 m/s)
lm = 0.50         # Min target distance — below this, ball doesn't reach
lM = 0.75         # Max target distance — above this, arm can't throw hard enough
gM = np.pi / 6    # Max lateral angle (±30°)
Ts = 0.02         # Simulation timestep
T  = 0.55         # Simulation horizon (ball lands by ~0.45s from z=0.5m at v≈0.9)
lc = 0.3          # Cost lengthscale — narrower than baseline (target range is 0.25m)
Nexp = 5          # Exploration throws
Nopt = 1500       # Policy optimisation steps per trial
M  = 400          # Monte Carlo particles
Nb = 250          # RBF basis functions

# Lengthscale rule: ls ≈ 0.15 × target_range
target_range = lM - lm  # 0.25m
lengthscales_init = [0.15 * target_range] * 2  # [0.0375, 0.0375]

# Release position (kuka_iiwa default)
RELEASE_POS = np.array([0.5, 0.0, 0.5])
LAUNCH_ANGLE_DEG = 35.0

# State/input dimensions
STATE_DIM = 8   # [x,y,z, vx,vy,vz, Px,Py]
INPUT_DIM = 1   # scalar speed
TARGET_DIM = 2  # [Px, Py]

# Results directory
mode_str = "coupled" if coupled else "decoupled"
log_dir = f"results_mc_pilot_pb_{mode_str}/{args.seed}"
os.makedirs(log_dir, exist_ok=True)

print(f"\n{'='*60}")
print(f"MC-PILOT PyBullet — {'COUPLED' if coupled else 'DECOUPLED'} MODE")
print(f"{'='*60}")
print(f"Seed: {args.seed}")
print(f"Trials: {args.num_trials}")
print(f"uM: {uM} m/s")
print(f"Target range: [{lm}, {lM}] m")
print(f"Cost lengthscale: {lc} m")
print(f"RBF lengthscales: {lengthscales_init}")
print(f"Coupled: {coupled}")
print(f"Log dir: {log_dir}")
print(f"{'='*60}\n")

# --- Target sampler ---
def sample_target():
    dist = np.random.uniform(lm, lM)
    angle = np.random.uniform(-gM, gM)
    return np.array([dist * np.cos(angle), dist * np.sin(angle)])

# --- Throwing system ---
throwing_system = PyBulletThrowingSystem(
    mass=0.0577,
    radius=0.0327,
    launch_angle_deg=LAUNCH_ANGLE_DEG,
    arm_noise=None,
    robot_name="kuka_iiwa",
    coupled=coupled,
    vel_ctrl_steps=10,
)

# --- RBF center initialization ---
# Centers cover the actual target sector
Px_lo = lm * np.cos(gM)
Px_hi = lM
Py_lo = lm * np.sin(-gM)
Py_hi = lM * np.sin(gM)
centers_init = np.random.uniform(
    [Px_lo, Py_lo], [Px_hi, Py_hi], size=(Nb, TARGET_DIM)
)
weight_init = np.random.uniform(-uM / 2, uM / 2, size=(1, Nb))

dtype = torch.float64
device = torch.device("cpu")

BALL_DIM = 6

# --- GP model learning ---
num_gp = 3
gp_input_dim = BALL_DIM

init_dict_RBF = {}
init_dict_RBF["active_dims"]            = np.arange(0, gp_input_dim)
init_dict_RBF["lengthscales_init"]      = np.ones(gp_input_dim)
init_dict_RBF["flg_train_lengthscales"] = True
init_dict_RBF["lambda_init"]            = np.ones(1)
init_dict_RBF["flg_train_lambda"]       = False
init_dict_RBF["sigma_n_init"]           = 1 * np.ones(1)
init_dict_RBF["flg_train_sigma_n"]      = True
init_dict_RBF["sigma_n_num"]            = None
init_dict_RBF["dtype"]                  = dtype
init_dict_RBF["device"]                 = device

model_learning_par = {}
model_learning_par["num_gp"]          = num_gp
model_learning_par["T_sampling"]      = Ts
model_learning_par["approximation_mode"] = "SOD"
model_learning_par["approximation_dict"] = {
    "SOD_threshold_mode": "relative",
    "SOD_threshold": 0.5,
    "flg_SOD_permutation": False,
}
model_learning_par["init_dict_list"]  = [init_dict_RBF] * num_gp
model_learning_par["dtype"]           = dtype
model_learning_par["device"]          = device

# --- MC_PILOT setup ---
mc_pilot = MC_PILCO.MC_PILOT(
    target_sampler=sample_target,
    release_position=RELEASE_POS,
    throwing_system=throwing_system,
    T_sampling=Ts,
    state_dim=STATE_DIM,
    input_dim=INPUT_DIM,
    f_model_learning=gp_model.Ballistic_Model_learning_RBF,
    model_learning_par=model_learning_par,
    f_rand_exploration_policy=Policy.Stratified_Throwing_Exploration,
    rand_exploration_policy_par={
        "full_state_dim": STATE_DIM,
        "u_max": uM,
        "n_strata": Nexp,
        "u_min": 0.0,
    },
    f_control_policy=Policy.Throwing_Policy,
    control_policy_par={
        "full_state_dim": STATE_DIM,
        "target_dim": TARGET_DIM,
        "num_basis": Nb,
        "u_max": uM,
        "lengthscales_init": lengthscales_init,
        "centers_init": centers_init,
        "weight_init": weight_init,
        "flg_drop": True,
    },
    f_cost_function=Cost_function.Throwing_Cost,
    cost_function_par={
        "position_indices": [0, 1],
        "target_indices":   [6, 7],
        "lengthscale":      lc,
        "dtype":            dtype,
        "device":           device,
    },
    std_meas_noise=np.zeros(STATE_DIM),
    log_path=log_dir,
    dtype=dtype,
    device=device,
    arm_noise=None,
)

# --- Policy optimisation dict ---
T_control = T
opt_steps = [Nopt] * (Nexp + args.num_trials)
lr_list = [0.01] * (Nexp + args.num_trials)

policy_optimization_dict = {
    "num_particles": M,
    "opt_steps_list": opt_steps,
    "lr_list": lr_list,
    "f_optimizer": "lambda p, lr : torch.optim.Adam(p, lr)",
    "num_step_print": 100,
    "policy_reinit_dict": {
        "lenghtscales_par": np.array(lengthscales_init),
        "centers_par": np.hstack([
            np.random.uniform(Px_lo, Px_hi, (Nb, 1)),
            np.random.uniform(Py_lo, Py_hi, (Nb, 1)),
        ]),
        "weight_par": uM / 2,
    },
    "p_dropout_list": [0.25] * (Nexp + args.num_trials),
    "alpha_cost": 0.99,
    "alpha_diff_cost": 0.99,
    "lr_reduction_ratio": 0.5,
    "lr_min": 0.0025,
    "p_drop_reduction": 0.125,
    "min_diff_cost": 0.02,
    "num_min_diff_cost": 400,
    "min_step": 400,
}

# --- Save config ---
config = {
    "uM": uM, "lm": lm, "lM": lM, "gM": gM, "Ts": Ts, "T": T,
    "lc": lc, "Nexp": Nexp, "Nopt": Nopt, "M": M, "Nb": Nb,
    "lengthscales_init": lengthscales_init,
    "release_pos": RELEASE_POS.tolist(),
    "launch_angle_deg": LAUNCH_ANGLE_DEG,
    "coupled": coupled,
    "robot_name": "kuka_iiwa",
    "seed": args.seed,
}
pkl.dump(config, open(os.path.join(log_dir, "config_log.pkl"), "wb"))

# --- Run ---
t_start = time.time()

model_optimization_opt_dict = {}
model_optimization_opt_dict["f_optimizer"]   = "lambda p : torch.optim.Adam(p, lr = 0.01)"
model_optimization_opt_dict["criterion"]     = Likelihood.Marginal_log_likelihood
model_optimization_opt_dict["N_epoch"]       = 1001
model_optimization_opt_dict["N_epoch_print"] = 500
model_optimization_opt_list = [model_optimization_opt_dict] * num_gp

cost_trial_list, _, _ = mc_pilot.reinforce(
    initial_state=np.zeros(STATE_DIM),      # Ignored by MC_PILOT
    initial_state_var=np.zeros(STATE_DIM),   # Ignored by MC_PILOT
    T_exploration=T,
    T_control=T_control,
    num_trials=args.num_trials,
    model_optimization_opt_list=model_optimization_opt_list,
    policy_optimization_dict=policy_optimization_dict,
    num_explorations=Nexp,
)

t_end = time.time()
print(f"\n{'='*60}")
print(f"Training complete in {t_end - t_start:.1f}s")

# --- Report results ---
noiseless = mc_pilot.noiseless_states_history
print(f"\n--- Results ({mode_str} mode) ---")
for i, traj in enumerate(noiseless):
    final = traj[-1]
    ball_xy = final[0:2]
    target_xy = final[6:8]
    err = np.linalg.norm(ball_xy - target_xy)
    phase = "Explore" if i < Nexp else f"Trial {i - Nexp + 1}"
    hit = "HIT" if err < 0.1 else ""
    tgt_dist = np.linalg.norm(target_xy)
    print(f"  Throw {i+1:2d} ({phase:>9s}): err={err:.3f}m  target={tgt_dist:.2f}m  {hit}")

# Count hits on policy throws only
policy_errors = []
for i in range(Nexp, len(noiseless)):
    final = noiseless[i][-1]
    err = np.linalg.norm(final[0:2] - final[6:8])
    policy_errors.append(err)

hits = sum(1 for e in policy_errors if e < 0.1)
print(f"\nPolicy throws: {hits}/{len(policy_errors)} hits")
print(f"Mean error: {np.mean(policy_errors):.3f}m")

if cost_trial_list:
    print(f"Final trial cost: {cost_trial_list[-1][-1]:.6f}")
