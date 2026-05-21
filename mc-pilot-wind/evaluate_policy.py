#!/usr/bin/env python3
import argparse
import os
import pickle as pkl
import sys
import numpy as np
import torch

import model_learning.Model_learning as ML
import model_learning.Model_learning_wind as ML_wind
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO as MC_PILCO_module
import policy_learning.Policy as Policy
from simulation_class.wind_models import ConstantWind, GustWind, TurbulentWind, WindModel

parser = argparse.ArgumentParser("Evaluate a trained MC-PILOT policy over new simulations")
parser.add_argument("--log_dir", type=str, required=True, help="Path to log dir, e.g. results_wind_W1/w5p0_aware/1")
parser.add_argument("--num_trials", type=int, default=100, help="Number of new random targets to evaluate")
parser.add_argument("--hit_threshold", type=float, default=0.1, help="Distance in meters to count as a hit")
args = parser.parse_args()

log_file_path = os.path.join(args.log_dir, "log.pkl")
config_file_path = os.path.join(args.log_dir, "config_log.pkl")

if not os.path.exists(log_file_path) or not os.path.exists(config_file_path):
    print(f"Cannot find log.pkl or config_log.pkl in {args.log_dir}")
    sys.exit(1)

config = pkl.load(open(config_file_path, "rb"))
wind_aware = config.get("wind_aware", False)
config_name = config.get("config", "")

print(f"==================================================")
print(f"Evaluating Policy: {args.log_dir}")
print(f"Wind Aware: {wind_aware}")
print(f"Config Name: {config_name}")
print(f"Trials: {args.num_trials}")
print(f"Hit Threshold: {args.hit_threshold*100:.0f} cm")
print(f"==================================================")

# Same parameters as training
seed = config["seed"] + 100 # Change seed so targets are new
np.random.seed(seed)
torch.manual_seed(seed)
dtype = torch.float64
device = torch.device("cpu")

Nb = config["Nb"]
uM = config["uM"]
Ts = config["Ts"]
T = config["T"]
lc = config["lc"]
lm = config["lm"]
lM = config["lM"]
gM = config.get("gM", np.pi / 6)

STATE_DIM = 10 if wind_aware else 8
INPUT_DIM  = 1
TARGET_DIM = 2
WIND_DIM   = 2

RELEASE_POS = np.array([0.0, 0.0, 0.5])

def sample_target():
    dist  = np.random.uniform(lm, lM)
    angle = np.random.uniform(-gM, gM)
    return np.array([dist * np.cos(angle), dist * np.sin(angle)])

# Reconstruct the wind model based on the config name
wind_speed = config.get("wind_speed", 0.0)
if "W1" in config_name or wind_speed > 0:
    wind_model = ConstantWind(velocity=[wind_speed, 0.0, 0.0])
elif "W2" in config_name:
    wind_model = GustWind(w_max=4.0)
elif "W3" in config_name:
    wind_model = TurbulentWind(w_mean=[2.5, 0.0, 0.0], sigma=4.0)
else:
    wind_model = WindModel()

from simulation_class.model_wind import WindThrowingSystem
throwing_system = WindThrowingSystem(
    mass=0.0577,
    radius=0.0327,
    launch_angle_deg=35.0,
    wind_model=wind_model,
    wind_aware=wind_aware,
)

# Setup dummy MC_PILOT object just to load the policy
if wind_aware:
    centers_init = np.zeros((Nb, 4))
    lengthscales_init = np.array([0.08, 0.08, 0.3, 0.3])
    f_control_policy = Policy.WindAware_Throwing_Policy
    control_policy_par = {
        "full_state_dim": STATE_DIM, "target_dim": TARGET_DIM, "wind_dim": WIND_DIM,
        "num_basis": Nb, "u_max": uM, "lengthscales_init": lengthscales_init,
        "centers_init": centers_init, "weight_init": np.zeros((1, Nb)),
        "flg_drop": True, "dtype": dtype, "device": device,
    }
else:
    centers_init = np.zeros((Nb, 2))
    lengthscales_init = np.array([0.08, 0.08])
    f_control_policy = Policy.Throwing_Policy
    control_policy_par = {
        "full_state_dim": STATE_DIM, "target_dim": TARGET_DIM,
        "num_basis": Nb, "u_max": uM, "lengthscales_init": lengthscales_init,
        "centers_init": centers_init, "weight_init": np.zeros((1, Nb)),
        "flg_drop": True, "dtype": dtype, "device": device,
    }

dummy_model_learning = ML.Ballistic_Model_learning_RBF(
    num_gp=3, T_sampling=Ts, approximation_mode="SOD", 
    approximation_dict={"SOD_threshold_mode": "relative", "SOD_threshold": 0.5, "flg_SOD_permutation": False},
    init_dict_list=[{"active_dims": np.arange(0, 6), "lengthscales_init": np.ones(6), "flg_train_lengthscales": True, "lambda_init": np.ones(1), "flg_train_lambda": False, "sigma_n_init": np.ones(1), "flg_train_sigma_n": True, "sigma_n_num": None, "dtype": dtype, "device": device}]*3,
    dtype=dtype, device=device
)

mc_pilot_obj = MC_PILCO_module.MC_PILOT_Wind(
    target_sampler=sample_target,
    release_position=RELEASE_POS,
    throwing_system=throwing_system,
    wind_model=wind_model,
    wind_aware=wind_aware,
    T_sampling=Ts,
    state_dim=STATE_DIM,
    input_dim=INPUT_DIM,
    f_model_learning=ML.Ballistic_Model_learning_RBF, # dummy
    model_learning_par={"num_gp": 3, "T_sampling": Ts, "approximation_mode": "SOD", "approximation_dict": {"SOD_threshold_mode": "relative", "SOD_threshold": 0.5, "flg_SOD_permutation": False}, "init_dict_list": [{"active_dims": np.arange(0, 6), "lengthscales_init": np.ones(6), "flg_train_lengthscales": True, "lambda_init": np.ones(1), "flg_train_lambda": False, "sigma_n_init": np.ones(1), "flg_train_sigma_n": True, "sigma_n_num": None, "dtype": dtype, "device": device}]*3, "dtype": dtype, "device": device},
    f_rand_exploration_policy=Policy.Stratified_Throwing_Exploration,
    rand_exploration_policy_par={"full_state_dim": STATE_DIM, "u_max": uM, "n_strata": 5, "dtype": dtype, "device": device},
    f_control_policy=f_control_policy,
    control_policy_par=control_policy_par,
    f_cost_function=Cost_function.Throwing_Cost,
    cost_function_par={"position_indices": [0, 1], "target_indices": [6, 7], "lengthscale": lc, "dtype": dtype, "device": device},
)

# Load parameters
log_dict = pkl.load(open(log_file_path, "rb"))
if "parameters_trial_list" not in log_dict or len(log_dict["parameters_trial_list"]) == 0:
    print("No trained trials found in log.pkl!")
    sys.exit(1)

num_trials_completed = len(log_dict["cost_trial_list"])
policy_params = log_dict["parameters_trial_list"][num_trials_completed-1]
mc_pilot_obj.control_policy.load_state_dict(policy_params)
print("Policy weights loaded successfully.\n")

np_policy = mc_pilot_obj.control_policy.get_np_policy()

errors = []
hits = 0

for i in range(args.num_trials):
    target = sample_target()
    s0 = np.concatenate([RELEASE_POS, np.zeros(3), target])
    
    if wind_aware:
        wind_model.reset()
        w0 = wind_model(0.0)
        s0 = np.concatenate([s0, w0[:2]])
        
    noisy_states, inputs, clean_states = throwing_system.rollout(s0=s0, policy=np_policy, T=T, dt=Ts, noise=np.zeros(STATE_DIM))
    
    # Calculate landing error
    landing_pos = clean_states[-1, 0:2]
    error = np.linalg.norm(landing_pos - target)
    errors.append(error)
    
    if error <= args.hit_threshold:
        hits += 1

mean_err = np.mean(errors) * 100
std_err = np.std(errors) * 100
hit_rate = (hits / args.num_trials) * 100

print(f"--- EVALUATION RESULTS ({args.num_trials} throws) ---")
print(f"Hit Rate (> {args.hit_threshold*100:.0f}cm) : {hit_rate:.1f}% ({hits}/{args.num_trials})")
print(f"Mean Error       : {mean_err:.2f} cm")
print(f"Std Error        : {std_err:.2f} cm")
print(f"==================================================")
