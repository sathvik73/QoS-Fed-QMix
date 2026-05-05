import os
import numpy as np
import matplotlib.pyplot as plt

def load_and_scale(data_dir, filename, target_len=600): # 30000 / 50 = 600
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"  WARNING: File not found: {path}")
        return None
    
    is_priority = "priority" in filename
    data = np.load(path, allow_pickle=is_priority)
    
    if len(data) < target_len:
        if is_priority:
            last_val = data[-1] if len(data) > 0 else {}
            data = list(data) + [last_val] * (target_len - len(data))
        else:
            last_val = data[-1] if len(data) > 0 else 0
            padding = np.full(target_len - len(data), last_val)
            data = np.concatenate([data, padding])
            
    return data[:target_len]

# Verified mapping of run folders to their result subfolders
run_paths = {
    "Model-Aided FedQMIX": "modelqmix_run/result/qmix/RDM_model_qmix",
    "Original QMIX": "original_run/result/qmix/RDM_qmix",
    "Proposed Priority-Aware": "proposed_run/result/qmix/RBM_proposed",
    "Baseline QMIX": "qmix_run/result/qmix/RBM_qmix"
}

target_episodes = 30000
evaluate_cycle = 50
target_len = target_episodes // evaluate_cycle
total_data_avail = 160000 # 10 devices * 16000

all_data = {}
for label, data_dir in run_paths.items():
    print(f"Loading {label}...")
    all_data[label] = {
        "total_collection": load_and_scale(data_dir, "episode_data_.npy", target_len),
        "rewards": load_and_scale(data_dir, "episode_rewards_.npy", target_len),
        "priority": load_and_scale(data_dir, "episode_priority_data_.npy", target_len)
    }

x_axis = np.arange(target_len) * evaluate_cycle
plot_dir = "comparison_plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

def save_plot(filename, title, ylabel, data_key, is_ratio=False, is_priority=None, scale='linear'):
    plt.figure(figsize=(10, 6))
    valid_plot = False
    for label, run_data in all_data.items():
        if is_priority is not None:
            if run_data["priority"] is not None:
                p_values = [d.get(float(is_priority), d.get(int(is_priority), 0)) for d in run_data["priority"]]
                y = np.array(p_values) / 16000.0 if is_ratio else np.array(p_values)
                plt.plot(x_axis, y, label=label, linewidth=2)
                valid_plot = True
        else:
            if run_data[data_key] is not None:
                y = run_data[data_key] / total_data_avail if is_ratio else run_data[data_key]
                plt.plot(x_axis, y, label=label, linewidth=2)
                valid_plot = True
                
    if not valid_plot:
        plt.close()
        return

    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{title} ({scale.capitalize()} Scale)", fontsize=14)
    plt.xscale(scale)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.close()

print("Generating plots...")

# 1. Total Data Collection Ratio
save_plot('total_data_ratio_normal.png', 'Total Data Collection Ratio', 'Ratio', 'total_collection', is_ratio=True, scale='linear')
save_plot('total_data_ratio_log.png', 'Total Data Collection Ratio', 'Ratio', 'total_collection', is_ratio=True, scale='log')

# 2. Total Rewards
save_plot('total_rewards_normal.png', 'Total Episode Reward', 'Reward', 'rewards', scale='linear')
save_plot('total_rewards_log.png', 'Total Episode Reward', 'Reward', 'rewards', scale='log')

# 3. Priority-wise Data collection ratios (1-10)
for p in range(1, 11):
    save_plot(f'priority_{p}_ratio_normal.png', f'Priority {p} Ratio', 'Ratio', 'priority', is_ratio=True, is_priority=p, scale='linear')
    save_plot(f'priority_{p}_ratio_log.png', f'Priority {p} Ratio', 'Ratio', 'priority', is_ratio=True, is_priority=p, scale='log')

print(f"Success! {len(os.listdir(plot_dir))} plots saved in {plot_dir}/")
