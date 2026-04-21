"""
MC-PILOT Config E-Strat3: platform height -0.5 m, targeted stratified + Nexp=10
AR525 Group-3, IIT Mandi

Same as E-Strat2 (targeted exploration [2.5, 3.5] m/s) but doubles Nexp to 10.
Root cause of E-Strat2 failure: GP data-starved. T=0.55s gives only 137 raw training
points before Trial 1 (vs 250 for D-Strat). GP simulation is inaccurate, optimizer
stalls at cost~0.25 even with good exploration throws.
Fix: Nexp=10 → 275 raw training points, comparable to D-Strat.
"""

import argparse
import os
import pickle as pkl

import numpy as np
import torch

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func
import model_learning.Model_learning as ML
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO as MC_PILCO_module
import policy_learning.Policy as Policy
import simulation_class.model as model_module

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser("test mc-pilot")
p.add_argument("-seed", type=int, default=1, help="random seed")
p.add_argument("-num_trials", type=int, default=10, help="number of learning trials")
locals().update(vars(p.parse_known_args()[0]))

torch.manual_seed(seed)
np.random.seed(seed)

dtype  = torch.float64
device = torch.device("cpu")
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Table 1 parameters
# ---------------------------------------------------------------------------
Nexp   = 10       # doubled from 5 — short T=0.55s gives only 27 steps/throw; need more data
Nopt   = 1500     # policy optimisation steps per trial
M      = 400      # Monte Carlo particles
Nb     = 250      # RBF basis functions in policy
uM     = 3.5      # max release speed (m/s)
uMin   = 2.5      # exploration floor — throws below this never reach lm=0.75m from z=0
Ts     = 0.02     # simulation timestep (s)
T      = 0.55     # simulation horizon (s) — ball lands in ~0.42s at max speed from z=0.0
lc     = 0.5      # cost lengthscale / success radius (m)
lm     = 0.75     # min target distance (m)
lM     = 1.10     # max target distance (m) — verified: max range from z=0.0 is ~1.16m with drag
gM     = np.pi / 6  # max lateral throw angle (rad)

# Augmented state dim: [x,y,z, vx,vy,vz, Px,Py] = 8
STATE_DIM  = 8
INPUT_DIM  = 1    # scalar release speed
BALL_DIM   = 6    # dims 0:6
TARGET_DIM = 2    # dims 6:8

RELEASE_POS = np.array([0.0, 0.0, 0.0])

# ---------------------------------------------------------------------------
# Target sampler
# ---------------------------------------------------------------------------
def sample_target():
    dist  = np.random.uniform(lm, lM)
    angle = np.random.uniform(-gM, gM)
    Px = dist * np.cos(angle)
    Py = dist * np.sin(angle)
    return np.array([Px, Py])

# ---------------------------------------------------------------------------
# Physics simulator
# ---------------------------------------------------------------------------
throwing_system = model_module.ThrowingSystem(
    mass=0.0577,
    radius=0.0327,
    launch_angle_deg=35.0,
    wind=None,
)

# ---------------------------------------------------------------------------
# GP model learning
# ---------------------------------------------------------------------------
num_gp      = 3
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

f_model_learning = ML.Ballistic_Model_learning_RBF

# ---------------------------------------------------------------------------
# Exploration policy — targeted stratification over [uMin, uM]
# ---------------------------------------------------------------------------
rand_exploration_policy_par = {
    "full_state_dim": STATE_DIM,
    "u_max": uM,
    "u_min": uMin,
    "n_strata": Nexp,   # 10 bands of 0.1 m/s each over [2.5, 3.5]
    "dtype": dtype,
    "device": device,
}
f_rand_exploration_policy = Policy.Stratified_Throwing_Exploration

# ---------------------------------------------------------------------------
# Control policy
# ---------------------------------------------------------------------------
centers_init = np.column_stack([
    np.random.uniform(lm * np.cos(-gM), lM, Nb),
    np.random.uniform(lm * np.sin(-gM), lM * np.sin(gM), Nb),
])
weight_init       = uM * (np.random.rand(1, Nb) - 0.5)
lengthscales_init = np.array([1.0, 1.0])

control_policy_par = {
    "full_state_dim": STATE_DIM,
    "target_dim": TARGET_DIM,
    "num_basis": Nb,
    "u_max": uM,
    "lengthscales_init": lengthscales_init,
    "centers_init": centers_init,
    "weight_init": weight_init,
    "flg_drop": True,
    "dtype": dtype,
    "device": device,
}
f_control_policy = Policy.Throwing_Policy

policy_reinit_dict = {
    "lenghtscales_par": lengthscales_init,
    "centers_par": np.array([1.0, 1.0]),
    "weight_par": uM,
}

# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------
cost_function_par = {
    "position_indices": [0, 1],
    "target_indices":   [6, 7],
    "lengthscale": lc,
    "dtype": dtype,
    "device": device,
}
f_cost_function = Cost_function.Throwing_Cost

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------
log_path = "results_mc_pilot_e_strat3/" + str(seed)
os.makedirs(log_path, exist_ok=True)

# ---------------------------------------------------------------------------
# Build MC_PILOT object
# ---------------------------------------------------------------------------
mc_pilot_obj = MC_PILCO_module.MC_PILOT(
    target_sampler=sample_target,
    release_position=RELEASE_POS,
    throwing_system=throwing_system,
    T_sampling=Ts,
    state_dim=STATE_DIM,
    input_dim=INPUT_DIM,
    f_model_learning=f_model_learning,
    model_learning_par=model_learning_par,
    f_rand_exploration_policy=f_rand_exploration_policy,
    rand_exploration_policy_par=rand_exploration_policy_par,
    f_control_policy=f_control_policy,
    control_policy_par=control_policy_par,
    f_cost_function=f_cost_function,
    cost_function_par=cost_function_par,
    std_meas_noise=1e-3 * np.ones(STATE_DIM),
    log_path=log_path,
    dtype=dtype,
    device=device,
)

# ---------------------------------------------------------------------------
# MC-PILCO options
# ---------------------------------------------------------------------------
model_optimization_opt_dict = {}
model_optimization_opt_dict["f_optimizer"]   = "lambda p : torch.optim.Adam(p, lr = 0.01)"
model_optimization_opt_dict["criterion"]     = Likelihood.Marginal_log_likelihood
model_optimization_opt_dict["N_epoch"]       = 1001
model_optimization_opt_dict["N_epoch_print"] = 500
model_optimization_opt_list = [model_optimization_opt_dict] * num_gp

policy_optimization_dict = {}
policy_optimization_dict["num_particles"]    = M
n_list = Nexp + num_trials
policy_optimization_dict["opt_steps_list"]   = [Nopt] * n_list
policy_optimization_dict["lr_list"]          = [0.01] * n_list
policy_optimization_dict["f_optimizer"]      = "lambda p, lr : torch.optim.Adam(p, lr)"
policy_optimization_dict["num_step_print"]   = 100
policy_optimization_dict["p_dropout_list"]   = [0.25] * n_list
policy_optimization_dict["p_drop_reduction"] = 0.25 / 2
policy_optimization_dict["alpha_diff_cost"]  = 0.99
policy_optimization_dict["min_diff_cost"]    = 0.02
policy_optimization_dict["num_min_diff_cost"]= 400
policy_optimization_dict["min_step"]         = 400
policy_optimization_dict["lr_min"]           = 0.0025
policy_optimization_dict["policy_reinit_dict"] = policy_reinit_dict

centre_target = np.array([lm + (lM - lm) / 2, 0.0])
initial_state = np.concatenate([RELEASE_POS, np.zeros(3), centre_target])
initial_state_var = np.concatenate([
    1e-4 * np.ones(6),
    (0.5 * (lM - lm)) ** 2 * np.ones(2),
])

reinforce_param_dict = {
    "initial_state":     initial_state,
    "initial_state_var": initial_state_var,
    "T_exploration":     T,
    "T_control":         T,
    "num_trials":        num_trials,
    "num_explorations":  Nexp,
    "model_optimization_opt_list": model_optimization_opt_list,
    "policy_optimization_dict":    policy_optimization_dict,
}

# ---------------------------------------------------------------------------
# Save config
# ---------------------------------------------------------------------------
config_log = {
    "seed": seed,
    "config": "E_strat3",
    "z_release": 0.0,
    "exploration": "stratified_targeted_nexp10",
    "u_min_exploration": uMin,
    "num_trials": num_trials,
    "Nexp": Nexp, "Nopt": Nopt, "M": M, "Nb": Nb,
    "uM": uM, "Ts": Ts, "T": T, "lc": lc,
    "lm": lm, "lM": lM, "gM": float(gM),
}
pkl.dump(config_log, open(log_path + "/config_log.pkl", "wb"))

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cost_trial_list, particles_states_list, particles_inputs_list = mc_pilot_obj.reinforce(
    **reinforce_param_dict
)

print("\n\nTraining complete.")
print(f"Final trial cost: {cost_trial_list[-1][-1]:.4f}")
