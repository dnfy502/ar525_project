import argparse
import os
import pickle as pkl
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12})

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def load_data(seed=1):
    result_dirs = {
        "W1-calm":     f"results_wind_W1/w0p0_blind/{seed}",
        "W1-light":    f"results_wind_W1/w2p5_blind/{seed}",
        "W1-moderate": f"results_wind_W1/w5p0_blind/{seed}",
        "W1-strong":   f"results_wind_W1/w8p0_blind/{seed}",
        "W1-aware":    f"results_wind_W1/w5p0_aware/{seed}",
        "W2-blind":    f"results_wind_W2/wmax4.0_blind/{seed}",
        "W2-aware":    f"results_wind_W2/wmax4.0_aware/{seed}",
        "W3-blind":    f"results_wind_W3/turb_s4.0_blind/{seed}",
        "W3-aware":    f"results_wind_W3/turb_s4.0_aware/{seed}",
    }
    
    data = {}
    for name, dirpath in result_dirs.items():
        log_file = os.path.join(dirpath, "log.pkl")
        cfg_file = os.path.join(dirpath, "config_log.pkl")
        
        if not os.path.exists(log_file):
            print(f"Missing {log_file}")
            continue
            
        with open(log_file, "rb") as f:
            log = pkl.load(f)
        with open(cfg_file, "rb") as f:
            cfg = pkl.load(f)
            
        nexp = cfg.get("Nexp", 5)
        
        noiseless = log.get("noiseless_states_history", [])
        costs = log.get("cost_trial_list", [])
        
        errors = []
        landings = []
        targets = []
        trial_errors = []
        
        for trial_idx in range(len(noiseless)):
            traj = noiseless[trial_idx]
            landing_xy = traj[-1, 0:2]
            target_xy  = traj[-1, 6:8]
            err = np.linalg.norm(landing_xy - target_xy)
            trial_errors.append(err)
            if trial_idx >= nexp:
                errors.append(err)
                landings.append(landing_xy)
                targets.append(target_xy)
                
        data[name] = {
            "log": log,
            "cfg": cfg,
            "nexp": nexp,
            "errors": np.array(errors),
            "trial_errors": np.array(trial_errors),
            "landings": np.array(landings),
            "targets": np.array(targets),
            "costs": costs
        }
    return data

def plot_bar_charts(data, outdir, hit_thresh=0.1):
    names = list(data.keys())
    hit_rates = []
    mean_errs = []
    
    for name in names:
        errors = data[name]["errors"]
        if len(errors) == 0:
            hit_rates.append(0)
            mean_errs.append(0)
            continue
        hits = np.sum(errors < hit_thresh)
        hit_rates.append(hits / len(errors) * 100)
        mean_errs.append(np.mean(errors) * 100) # cm
        
    # Hit rates
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=names, y=hit_rates, ax=ax, palette="viridis")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title(f"Hit Rates by Configuration (Threshold={hit_thresh*100:.0f} cm)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hit_rates.png"), dpi=300)
    plt.close()
    
    # Mean errors
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=names, y=mean_errs, ax=ax, palette="rocket")
    ax.set_ylabel("Mean Error (cm)")
    ax.set_title("Mean Landing Error by Configuration")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mean_errors.png"), dpi=300)
    plt.close()

def plot_learning_curves(data, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    groups = {
        "W1 (Constant Wind)": ["W1-calm", "W1-light", "W1-moderate", "W1-strong", "W1-aware"],
        "W2 (Gusts)": ["W2-blind", "W2-aware"],
        "W3 (Turbulence)": ["W3-blind", "W3-aware"],
        "Aware vs Blind (Hard Conditions)": ["W1-moderate", "W1-aware", "W2-blind", "W2-aware", "W3-blind", "W3-aware"]
    }
    
    for ax, (title, names) in zip(axes, groups.items()):
        for name in names:
            if name not in data: continue
            trial_errs = data[name]["trial_errors"] * 100 # cm
            ax.plot(range(1, len(trial_errs) + 1), trial_errs, marker='o', label=name)
            # Add vertical line for Nexp
            nexp = data[name]["nexp"]
            ax.axvline(nexp, color='grey', linestyle='--', alpha=0.3)
            
        ax.set_title(title)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Landing Error (cm)")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "learning_curves_error.png"), dpi=300)
    plt.close()
    
    # Plot Costs
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for ax, (title, names) in zip(axes, groups.items()):
        for name in names:
            if name not in data: continue
            costs = data[name]["costs"]
            final_costs = [c[-1] for c in costs if len(c) > 0]
            if not final_costs: continue
            # PAD final costs if needed (starts after Nexp typically)
            x_vals = range(data[name]['nexp'] + 1, data[name]['nexp'] + 1 + len(final_costs))
            ax.plot(x_vals, final_costs, marker='s', label=name)
            
        ax.set_title(title + " - Policy Cost")
        ax.set_xlabel("Optimization Iteration (Trial)")
        ax.set_ylabel("Expected Cost")
        ax.legend()
        ax.set_yscale('log')
        
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "learning_curves_cost.png"), dpi=300)
    plt.close()

def plot_error_distributions(data, outdir):
    all_errors = []
    labels = []
    for name, d in data.items():
        if len(d["errors"]) > 0:
            all_errors.extend(d["errors"] * 100)
            labels.extend([name] * len(d["errors"]))
            
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=labels, y=all_errors, ax=ax, palette="Set3")
    ax.set_ylabel("Landing Error (cm)")
    ax.set_title("Distribution of Post-Exploration Landing Errors")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "error_distributions_boxplot.png"), dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(x=labels, y=all_errors, ax=ax, palette="Set3", inner="point")
    ax.set_ylabel("Landing Error (cm)")
    ax.set_title("Violin Plot of Post-Exploration Landing Errors")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "error_distributions_violin.png"), dpi=300)
    plt.close()

def plot_2d_scatter(data, outdir):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for ax, name in zip(axes, list(data.keys())):
        d = data[name]
        if len(d["landings"]) == 0:
            ax.set_title(f"{name} - No Data")
            continue
            
        landings = d["landings"]
        targets = d["targets"]
        
        # We can center the targets to (0,0) to show relative error more clearly
        # or plot absolute positions. Let's plot absolute to see target distribution
        ax.scatter(targets[:, 0], targets[:, 1], c='red', marker='x', s=100, label='Target')
        ax.scatter(landings[:, 0], landings[:, 1], c='blue', alpha=0.6, label='Landing')
        
        for t, l in zip(targets, landings):
            ax.plot([t[0], l[0]], [t[1], l[1]], 'k-', alpha=0.2)
            
        ax.set_title(name)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect('equal', adjustable='datalim')
        if name == list(data.keys())[0]:
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scatter_2d_landings.png"), dpi=300)
    plt.close()
    
def plot_wind_profiles(data, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    names_to_plot = ["W1-aware", "W2-aware", "W3-aware"]
    for i, name in enumerate(names_to_plot):
        if name not in data: continue
        ax = axes[i]
        log = data[name]["log"]
        noiseless = log.get("noiseless_states_history", [])
        if len(noiseless) == 0: continue
        # Take the last trajectory
        traj = noiseless[-1]
        if traj.shape[1] >= 10: # Aware has 10 states (0-9)
            wind_x = traj[:, 8]
            wind_y = traj[:, 9]
            dt = 0.05 # Assuming standard dt, not strictly needed for relative shape
            time = np.arange(len(wind_x)) * dt
            ax.plot(time, wind_x, label="Wind X (m/s)")
            ax.plot(time, wind_y, label="Wind Y (m/s)")
            ax.set_title(f"{name} - Final Trial Wind Profile")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Wind Speed (m/s)")
            ax.legend()
    
    # For blind models, wind is not in the state, so we might not be able to plot it directly from state
    # But let's plot W3-blind to see if state has it? No, blind is 8D.
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "wind_profiles.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hit_threshold", type=float, default=0.1)
    args = parser.parse_args()
    
    outdir = "performance_analysis_plots"
    ensure_dir(outdir)
    
    print("Loading data...")
    data = load_data(args.seed)
    
    print("Generating bar charts...")
    plot_bar_charts(data, outdir, args.hit_threshold)
    
    print("Generating learning curves...")
    plot_learning_curves(data, outdir)
    
    print("Generating error distributions...")
    plot_error_distributions(data, outdir)
    
    print("Generating 2D scatter plots...")
    plot_2d_scatter(data, outdir)
    
    print("Generating wind profiles...")
    plot_wind_profiles(data, outdir)
    
    print(f"All plots saved to {os.path.abspath(outdir)}")

if __name__ == "__main__":
    main()
