"""
Quick max-range verification for each platform height config.
Throws at max speed (uM=3.5, alpha=35deg) straight ahead (Py=0) and
measures actual landing distance from ThrowingSystem (with drag).
"""
import sys
import numpy as np
sys.path.insert(0, ".")
import simulation_class.model as model_module

uM    = 3.5
alpha = 35.0
mass  = 0.0577
radius = 0.0327
Ts    = 0.02

throwing_system = model_module.ThrowingSystem(
    mass=mass, radius=radius, launch_angle_deg=alpha, wind=None
)

configs = {
    "A": 0.5,
    "B": 1.0,
    "C": 1.5,
    "D": 2.0,
    "E": 0.0,
}

print(f"{'Config':>8} {'z_release':>10} {'landing_x':>12} {'flight_t':>10}")
print("-" * 45)

for name, z_rel in configs.items():
    release_pos = np.array([0.0, 0.0, z_rel])
    # Target far away so policy returns max speed
    far_target  = np.array([99.0, 0.0])
    s0 = np.concatenate([release_pos, np.zeros(3), far_target])

    # Policy always returns uM
    def max_speed_policy(state, t):
        return np.array([uM])

    T_sim = 3.0  # long enough for any config
    noisy, inputs, clean = throwing_system.rollout(
        s0=s0,
        policy=max_speed_policy,
        T=T_sim,
        dt=Ts,
        noise=np.zeros(8),
    )
    land_x  = clean[-1, 0]
    land_z  = clean[-1, 2]
    n_steps = clean.shape[0]
    flight_t = (n_steps - 1) * Ts
    print(f"{'  '+name:>8} {z_rel:>10.1f} {land_x:>12.4f} {flight_t:>10.3f}s  (z_land={land_z:.4f})")
