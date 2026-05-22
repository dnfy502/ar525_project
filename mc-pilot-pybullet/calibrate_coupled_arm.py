"""
Calibrate coupled arm: measure v_cmd → v_actual mapping.

For each commanded speed, plans a throw, executes the arm trajectory with
VELOCITY_CONTROL in the final steps, releases the ball in coupled mode
(no resetBaseVelocity), and measures:
  1. Actual ball velocity at release
  2. Landing range

Usage:
  python calibrate_coupled_arm.py
  python calibrate_coupled_arm.py --robot kuka_iiwa --vel_ctrl_steps 10
  python calibrate_coupled_arm.py --robot franka_panda
"""

import argparse
import sys

import numpy as np
import pybullet as p
import pybullet_data

sys.path.insert(0, ".")
from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import available_robot_names, get_robot_profile
from simulation_class.model import _ball_accel

parser = argparse.ArgumentParser("Coupled arm velocity calibration")
parser.add_argument("--robot", type=str, default="kuka_iiwa",
                    choices=available_robot_names())
parser.add_argument("--vel_ctrl_steps", type=int, default=10,
                    help="number of steps using VELOCITY_CONTROL before release")
parser.add_argument("--speed_min", type=float, default=0.5)
parser.add_argument("--speed_max", type=float, default=3.5)
parser.add_argument("--speed_step", type=float, default=0.25)
parser.add_argument("--launch_angle", type=float, default=35.0,
                    help="launch angle in degrees")
parser.add_argument("--target_dist", type=float, default=1.0,
                    help="target distance for azimuth calculation")
args = parser.parse_args()

profile = get_robot_profile(args.robot)
BALL_MASS = 0.0577
BALL_RADIUS = 0.0327
LAUNCH_ANGLE = np.deg2rad(args.launch_angle)
DT = 0.02
T_W, T_R, T_ARM = profile.timing
RELEASE_POS = np.array(profile.default_release_pos, dtype=float)
TARGET = np.array([args.target_dist, 0.0])

# Azimuth from release to target
phi = np.arctan2(TARGET[1] - RELEASE_POS[1], TARGET[0] - RELEASE_POS[0])


def speed_to_velocity(speed):
    return np.array([
        speed * np.cos(LAUNCH_ANGLE) * np.cos(phi),
        speed * np.cos(LAUNCH_ANGLE) * np.sin(phi),
        speed * np.sin(LAUNCH_ANGLE),
    ])


def run_one_throw(speed, vel_ctrl_steps):
    """Run a single coupled throw and return (v_actual_3d, landing_range)."""
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(DT, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)

    urdf_path = pybullet_data.getDataPath() + "/" + profile.urdf_rel_path
    arm = ArmController(client, urdf_path, robot_name=profile.name)
    arm.reset()

    ee_pos, _, _, _ = arm.ee_state()
    ball_col = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS,
                                       physicsClientId=client)
    ball_vis = p.createVisualShape(p.GEOM_SPHERE, radius=BALL_RADIUS,
                                    rgbaColor=[1, 1, 0, 1],
                                    physicsClientId=client)
    ball_id = p.createMultiBody(baseMass=BALL_MASS,
                                 baseCollisionShapeIndex=ball_col,
                                 baseVisualShapeIndex=ball_vis,
                                 basePosition=ee_pos.tolist(),
                                 physicsClientId=client)
    p.changeDynamics(ball_id, -1, linearDamping=0.0, angularDamping=0.0,
                     physicsClientId=client)

    v_cmd = speed_to_velocity(speed)
    arm.attach_ball(ball_id, coupled=True)
    t_arm = max(T_ARM, T_R + 0.5)
    coeffs, _, qd_release, v_achieved = arm.plan_throw(
        v_cmd, RELEASE_POS, t_w=T_W, t_r=T_R, T=t_arm
    )

    release_step = int(T_R / DT)
    released = False
    actual_vel = None
    total_steps = int(t_arm / DT) + int(1.5 / DT)  # arm + flight

    pos_traj = []

    for step in range(total_steps):
        t = step * DT
        if not released:
            q_t, qd_t = arm.get_setpoint(coeffs, t)
            # Boost forces near release for tighter tracking
            if step >= release_step - vel_ctrl_steps:
                arm.step_velocity(q_t, qd_t)
            else:
                arm.step(q_t, qd_t)

            if step >= release_step:
                actual_vel = arm.release_ball_coupled(ball_id)
                released = True
                ball_pos, _ = p.getBasePositionAndOrientation(
                    ball_id, physicsClientId=client)
                pos_traj.append(np.array(ball_pos))
        else:
            ball_pos, _ = p.getBasePositionAndOrientation(
                ball_id, physicsClientId=client)
            ball_vel, _ = p.getBaseVelocity(ball_id, physicsClientId=client)
            pos = np.array(ball_pos)
            vel = np.array(ball_vel)

            a_total = _ball_accel(pos, vel, BALL_MASS, BALL_RADIUS,
                                   np.zeros(3))
            a_drag = a_total - np.array([0.0, 0.0, -9.81])
            f_drag = BALL_MASS * a_drag
            p.applyExternalForce(ball_id, -1, f_drag.tolist(), [0, 0, 0],
                                  p.WORLD_FRAME, physicsClientId=client)

            pos_traj.append(pos.copy())

            if pos[2] <= BALL_RADIUS + 0.005 and len(pos_traj) > 3:
                # Interpolate landing
                prev_pos = pos_traj[-2]
                if prev_pos[2] > 0:
                    frac = prev_pos[2] / (prev_pos[2] - pos[2])
                    land = prev_pos + frac * (pos - prev_pos)
                    land[2] = 0.0
                    pos_traj[-1] = land
                break

        p.stepSimulation(physicsClientId=client)

    p.disconnect(client)

    if pos_traj:
        landing = pos_traj[-1]
        landing_range = np.sqrt(landing[0]**2 + landing[1]**2)
    else:
        landing_range = 0.0

    return actual_vel, landing_range


# Also run decoupled comparison
def run_one_throw_decoupled(speed):
    """Run a single decoupled throw (set_vel=v_cmd) for comparison."""
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(DT, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)

    urdf_path = pybullet_data.getDataPath() + "/" + profile.urdf_rel_path
    arm = ArmController(client, urdf_path, robot_name=profile.name)
    arm.reset()

    ee_pos, _, _, _ = arm.ee_state()
    ball_col = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS,
                                       physicsClientId=client)
    ball_vis = p.createVisualShape(p.GEOM_SPHERE, radius=BALL_RADIUS,
                                    rgbaColor=[1, 1, 0, 1],
                                    physicsClientId=client)
    ball_id = p.createMultiBody(baseMass=BALL_MASS,
                                 baseCollisionShapeIndex=ball_col,
                                 baseVisualShapeIndex=ball_vis,
                                 basePosition=ee_pos.tolist(),
                                 physicsClientId=client)
    p.changeDynamics(ball_id, -1, linearDamping=0.0, angularDamping=0.0,
                     physicsClientId=client)

    v_cmd = speed_to_velocity(speed)
    arm.attach_ball(ball_id)
    t_arm = max(T_ARM, T_R + 0.5)
    coeffs, _, _, _ = arm.plan_throw(v_cmd, RELEASE_POS, t_w=T_W, t_r=T_R, T=t_arm)

    release_step = int(T_R / DT)
    released = False
    total_steps = int(t_arm / DT) + int(1.5 / DT)
    pos_traj = []

    for step in range(total_steps):
        t = step * DT
        if not released:
            q_t, qd_t = arm.get_setpoint(coeffs, t)
            arm.step(q_t, qd_t)
            if step >= release_step:
                arm.release_ball(ball_id, set_vel=v_cmd)
                released = True
                ball_pos, _ = p.getBasePositionAndOrientation(
                    ball_id, physicsClientId=client)
                pos_traj.append(np.array(ball_pos))
        else:
            ball_pos, _ = p.getBasePositionAndOrientation(
                ball_id, physicsClientId=client)
            ball_vel, _ = p.getBaseVelocity(ball_id, physicsClientId=client)
            pos = np.array(ball_pos)
            vel = np.array(ball_vel)

            a_total = _ball_accel(pos, vel, BALL_MASS, BALL_RADIUS,
                                   np.zeros(3))
            a_drag = a_total - np.array([0.0, 0.0, -9.81])
            f_drag = BALL_MASS * a_drag
            p.applyExternalForce(ball_id, -1, f_drag.tolist(), [0, 0, 0],
                                  p.WORLD_FRAME, physicsClientId=client)
            pos_traj.append(pos.copy())

            if pos[2] <= BALL_RADIUS + 0.005 and len(pos_traj) > 3:
                prev_pos = pos_traj[-2]
                if prev_pos[2] > 0:
                    frac = prev_pos[2] / (prev_pos[2] - pos[2])
                    land = prev_pos + frac * (pos - prev_pos)
                    land[2] = 0.0
                    pos_traj[-1] = land
                break

        p.stepSimulation(physicsClientId=client)

    p.disconnect(client)

    if pos_traj:
        landing = pos_traj[-1]
        return np.sqrt(landing[0]**2 + landing[1]**2)
    return 0.0


# --- Run calibration sweep ---
speeds = np.arange(args.speed_min, args.speed_max + 0.01, args.speed_step)

print(f"\nCoupled Arm Calibration — {profile.name}")
print(f"Release pos: {RELEASE_POS}")
print(f"Launch angle: {args.launch_angle}°")
print(f"VEL_CTRL steps: {args.vel_ctrl_steps}")
print(f"Timing: t_w={T_W}, t_r={T_R}, T_arm={T_ARM}")
print(f"Joint velocity limits: {profile.qd_max}")
print()

header = (f"{'v_cmd':>7s}  {'|v_actual|':>10s}  {'ratio':>6s}  "
          f"{'range_cpl':>10s}  {'range_dec':>10s}  "
          f"{'v_actual':>30s}  {'clip':>5s}")
print(header)
print("-" * len(header))

results = []
for speed in speeds:
    v_actual, range_cpl = run_one_throw(speed, args.vel_ctrl_steps)
    range_dec = run_one_throw_decoupled(speed)

    if v_actual is not None:
        v_mag = np.linalg.norm(v_actual)
        ratio = v_mag / speed if speed > 0 else 0.0
        v_str = f"[{v_actual[0]:+.3f}, {v_actual[1]:+.3f}, {v_actual[2]:+.3f}]"
        # Check if joint velocity limits were hit
        clipped = "yes" if ratio < 0.85 else "no"
    else:
        v_mag = 0.0
        ratio = 0.0
        v_str = "[N/A]"
        clipped = "N/A"

    results.append({
        "v_cmd": speed, "v_actual_mag": v_mag, "ratio": ratio,
        "range_coupled": range_cpl, "range_decoupled": range_dec,
        "clipped": clipped,
    })
    print(f"{speed:7.2f}  {v_mag:10.3f}  {ratio:6.3f}  "
          f"{range_cpl:10.3f}  {range_dec:10.3f}  "
          f"{v_str:>30s}  {clipped:>5s}")

print()
# Recommend uM
usable = [r for r in results if r["ratio"] > 0.80]
if usable:
    max_usable = max(usable, key=lambda r: r["v_cmd"])
    print(f"Recommended uM (coupled):  {max_usable['v_cmd']:.2f} m/s  "
          f"(delivers {max_usable['v_actual_mag']:.3f} m/s, "
          f"ratio={max_usable['ratio']:.3f})")
    print(f"  Achievable range: {max_usable['range_coupled']:.3f} m")

    # Recommend lm/lM from coupled range
    min_range = results[0]["range_coupled"] if results[0]["ratio"] > 0.5 else results[1]["range_coupled"]
    max_range = max_usable["range_coupled"]
    print(f"  Suggested lm: {min_range:.2f} m")
    print(f"  Suggested lM: {max_range:.2f} m")
else:
    print("WARNING: No commanded speed achieved >80% velocity tracking.")
    print("The arm cannot deliver meaningful velocities in coupled mode.")
    print("Consider using a different robot or relaxing joint velocity limits.")
