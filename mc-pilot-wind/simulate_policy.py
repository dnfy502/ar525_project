#!/usr/bin/env python3
import argparse
import os
import pickle as pkl
import time
import sys
import numpy as np
import torch

import model_learning.Model_learning as ML
import model_learning.Model_learning_wind as ML_wind
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO as MC_PILCO_module
import policy_learning.Policy as Policy
from simulation_class.wind_models import ConstantWind, WindModel

sys.path.append("../mc-pilot-pybullet")
from simulation_class.model_pybullet import PyBulletThrowingSystem

# Inject time.sleep to slow down pybullet GUI for visualization
_original_simulate = PyBulletThrowingSystem._simulate_pybullet

def _slow_simulate_pybullet(self, release_pos, v_cmd, T, dt):
    # Call original but slow it down slightly
    # PyBullet GUI without real-time needs sleep to be viewable
    import pybullet as p
    
    mode = p.GUI if self._gui_mode else p.DIRECT
    client = p.connect(mode)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(dt, physicsClientId=client)
    p.setAdditionalSearchPath(importlib_data_path(), physicsClientId=client)
    p.loadURDF(self._plane_urdf, physicsClientId=client)

    # We will just let the original run, but monkey patching the p.stepSimulation inside is hard.
    # Instead, we just let it run. In PyBullet GUI, even without sleep, the rendering takes a moment.
    # But let's actually just define a safe wrapper.
    pass

# We will just patch p.stepSimulation globally for this script if we want, but let's just use a simple approach.
import pybullet as p
_original_stepSimulation = p.stepSimulation
def _slow_stepSimulation(*args, **kwargs):
    time.sleep(0.01) # slow down 
    return _original_stepSimulation(*args, **kwargs)
p.stepSimulation = _slow_stepSimulation

parser = argparse.ArgumentParser("Simulate trained MC-PILOT Wind policy")
parser.add_argument("--log_dir", type=str, required=True, help="Directory containing log.pkl (e.g. results_wind_W1/w5p0_aware/1)")
parser.add_argument("--num_trials", type=int, default=5, help="Number of visual throws")
args = parser.parse_args()

log_file_path = os.path.join(args.log_dir, "log.pkl")
config_file_path = os.path.join(args.log_dir, "config_log.pkl")

if not os.path.exists(log_file_path) or not os.path.exists(config_file_path):
    print(f"Cannot find log.pkl or config_log.pkl in {args.log_dir}")
    sys.exit(1)

config = pkl.load(open(config_file_path, "rb"))
wind_aware = config.get("wind_aware", False)
wind_speed = config.get("wind_speed", 0.0)

print(f"Loading configuration from {args.log_dir}")
print(f"Wind Aware: {wind_aware}, Wind Speed: {wind_speed}")

seed = config["seed"]
torch.manual_seed(seed)
np.random.seed(seed)
dtype = torch.float64
device = torch.device("cpu")

Nexp = config["Nexp"]
M = config["M"]
Nb = config["Nb"]
uM = config["uM"]
Ts = config["Ts"]
T = config["T"]
lc = config["lc"]
lm = config["lm"]
lM = config["lM"]
gM = config["gM"]

STATE_DIM = 10 if wind_aware else 8
INPUT_DIM  = 1
BALL_DIM   = 6
TARGET_DIM = 2
WIND_DIM   = 2

RELEASE_POS = np.array([0.0, 0.0, 0.5])

def sample_target():
    dist  = np.random.uniform(lm, lM)
    angle = np.random.uniform(-gM, gM)
    return np.array([dist * np.cos(angle), dist * np.sin(angle)])

if wind_speed > 0:
    wind_model = ConstantWind(velocity=[wind_speed, 0.0, 0.0])
else:
    wind_model = WindModel()

throwing_system = PyBulletThrowingSystem(
    mass=0.0577,
    radius=0.0327,
    launch_angle_deg=35.0,
    wind_model=wind_model,
    wind_aware=wind_aware,
    gui_mode=True, # <--- ENABLING GUI
)

if wind_aware:
    f_model_learning = ML_wind.WindAware_Ballistic_Model_learning_RBF
    f_control_policy = Policy.WindAware_Throwing_Policy
    def wind_sampler():
        return np.array([wind_speed, 0.0])
else:
    f_model_learning = ML.Ballistic_Model_learning_RBF
    f_control_policy = Policy.Throwing_Policy
    wind_sampler = None

# We must initialize the policy with the exact same structure to load weights
if wind_aware:
    centers_init = np.zeros((Nb, 4))
    lengthscales_init = np.array([0.08, 0.08, 0.3, 0.3])
    control_policy_par = {
        "full_state_dim": STATE_DIM, "target_dim": TARGET_DIM, "wind_dim": WIND_DIM,
        "num_basis": Nb, "u_max": uM, "lengthscales_init": lengthscales_init,
        "centers_init": centers_init, "weight_init": np.zeros((1, Nb)),
        "flg_drop": True, "dtype": dtype, "device": device,
    }
else:
    centers_init = np.zeros((Nb, 2))
    lengthscales_init = np.array([0.08, 0.08])
    control_policy_par = {
        "full_state_dim": STATE_DIM, "target_dim": TARGET_DIM,
        "num_basis": Nb, "u_max": uM, "lengthscales_init": lengthscales_init,
        "centers_init": centers_init, "weight_init": np.zeros((1, Nb)),
        "flg_drop": True, "dtype": dtype, "device": device,
    }

mc_pilot_obj = MC_PILCO_module.MC_PILOT_Wind(
    target_sampler=sample_target,
    release_position=RELEASE_POS,
    throwing_system=throwing_system,
    wind_model=wind_model,
    wind_aware=wind_aware,
    T_sampling=Ts,
    state_dim=STATE_DIM,
    input_dim=INPUT_DIM,
    f_model_learning=f_model_learning,
    model_learning_par={"num_gp": 3, "T_sampling": Ts, "approximation_mode": "SOD", "approximation_dict": {"SOD_threshold_mode": "relative", "SOD_threshold": 0.5, "flg_SOD_permutation": False}, "init_dict_list": [{"active_dims": np.arange(0, 8 if wind_aware else 6), "lengthscales_init": np.ones(8 if wind_aware else 6), "flg_train_lengthscales": True, "lambda_init": np.ones(1), "flg_train_lambda": False, "sigma_n_init": np.ones(1), "flg_train_sigma_n": True, "sigma_n_num": None, "dtype": dtype, "device": device}] * 3, "dtype": dtype, "device": device},
    f_rand_exploration_policy=Policy.Stratified_Throwing_Exploration,
    rand_exploration_policy_par={"full_state_dim": STATE_DIM, "u_max": uM, "n_strata": Nexp, "dtype": dtype, "device": device},
    f_control_policy=f_control_policy,
    control_policy_par=control_policy_par,
    f_cost_function=Cost_function.Throwing_Cost,
    cost_function_par={"position_indices": [0, 1], "target_indices": [6, 7], "lengthscale": lc, "dtype": dtype, "device": device},
    std_meas_noise=1e-3 * np.ones(STATE_DIM),
    log_path=args.log_dir,
    dtype=dtype,
    device=device,
    wind_sampler=wind_sampler,
)

# Load parameters
log_dict = pkl.load(open(log_file_path, "rb"))
num_trials_completed = len(log_dict.get("cost_trial_list", []))
if num_trials_completed == 0:
    print("No trained trials found in log.pkl!")
    sys.exit(1)

print(f"Loading parameters from trial {num_trials_completed}")
mc_pilot_obj.load_model_from_log(num_trial=num_trials_completed, num_explorations=Nexp, folder=args.log_dir + "/")

policy_params = log_dict["parameters_trial_list"][num_trials_completed-1]
mc_pilot_obj.control_policy.load_state_dict(policy_params)
print("Policy successfully loaded.")

np_policy = mc_pilot_obj.control_policy.get_np_policy()

for i in range(args.num_trials):
    print(f"\nSimulation {i+1}/{args.num_trials}")
    target = sample_target()
    
    # Target visual indicator in PyBullet? We can't easily draw it inside throwing_system rollout
    # but we can at least observe the throw.
    print(f"Target: {target}")
    
    s0 = np.concatenate([RELEASE_POS, np.zeros(3), target])
    if wind_aware:
        wind_model.reset()
        w0 = wind_model(0.0)
        s0 = np.concatenate([s0, w0[:2]])
        
    print("Rolling out...")
    mc_pilot_obj.system.rollout(s0=s0, policy=np_policy, T=T, dt=Ts, noise=1e-3)
    time.sleep(1) # brief pause between throws

print("Simulations complete!")
