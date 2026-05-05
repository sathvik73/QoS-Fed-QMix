import numpy as np
import matplotlib.pyplot as plt
import os
import collections

base_dir = r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting\new_runs"
save_dir = r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting\result\new_paper_plots"
os.makedirs(save_dir, exist_ok=True)

runs = [
    {
        "label": "Proposed FedQMIX (Ours)",
        "folder": "Proposed",
        "result_path": "result/qmix/RDM_proposedf",
        "c": "red", "ls": "-", "marker": "o", "is_fed": True
    },
    {
        "label": "Original FedQMIX [4]",
        "folder": "Original",
        "result_path": "my_project/result/qmix/RDM_original",
        "c": "darkorange", "ls": "-", "marker": "s", "is_fed": True
    },
    {
        "label": "Model-Aided QMIX [4]",
        "folder": "Model-qmix",
        "result_path": "my_project/result/qmix/RDM_model_qmix",
        "c": "dodgerblue", "ls": "-", "marker": "^", "is_fed": False
    },
    {
        "label": "Standard QMIX [3]",
        "folder": "qmix",
        "result_path": "my_project/result/qmix/RDM_qmix",
        "c": "green", "ls": "-", "marker": "*", "is_fed": False
    }
]

def smooth(y, box_pts=12):
    if len(y) < box_pts: return y
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return list(y[:box_pts-1]) + list(y_smooth)

data_dict = {}

# Target episodes and cycle
EVAL_CYCLE = 50
MAX_EPISODES = 30000
MAX_IDX = (MAX_EPISODES // EVAL_CYCLE) + 1 # Up to 601 indices

for run in runs:
    sub_dir = os.path.join(base_dir, run["folder"], run["result_path"])
    if not os.path.exists(sub_dir):
        print(f"Result folder not found for {run['label']} at {sub_dir}")
        continue
    
    is_fed = run["is_fed"]
    f_total = os.path.join(sub_dir, "global_data.npy") if is_fed else os.path.join(sub_dir, "episode_data_.npy")
    f_rew = os.path.join(sub_dir, "global_rewards.npy") if is_fed else os.path.join(sub_dir, "episode_rewards_.npy")
    if is_fed:
        p_data = []
        for agent_idx in range(3):
            f_pri_agent = os.path.join(sub_dir, f"episode_priority_data_{agent_idx}.npy")
            if os.path.exists(f_pri_agent):
                p_data.append(list(np.load(f_pri_agent, allow_pickle=True))[:MAX_IDX])
        priority_data = []
        if len(p_data) > 0:
            for ep_idx in range(len(p_data[0])):
                ep_dict = collections.defaultdict(float)
                for agent_data in p_data:
                    if ep_idx < len(agent_data) and isinstance(agent_data[ep_idx], dict):
                        for k, v in agent_data[ep_idx].items():
                            ep_dict[k] += v
                priority_data.append(dict(ep_dict))
    else:
        f_pri = os.path.join(sub_dir, "episode_priority_data_.npy")
        priority_data = list(np.load(f_pri, allow_pickle=True))[:MAX_IDX] if os.path.exists(f_pri) else []
    
    total_data = list(np.load(f_total, allow_pickle=True))[:MAX_IDX] if os.path.exists(f_total) else []
    reward_data = list(np.load(f_rew, allow_pickle=True))[:MAX_IDX] if os.path.exists(f_rew) else []
    
    metrics = {
        "total": total_data,
        "reward": reward_data,
        "priority": priority_data,
        "run_cfg": run
    }
    
    if len(metrics["total"]) > 0:
        data_dict[run["label"]] = metrics

# Find maximums
MAX_TOTAL = 0
priorities_found = set()

for lbl, metrics in data_dict.items():
    if len(metrics["total"]) > 0:
        MAX_TOTAL = max(MAX_TOTAL, np.max(metrics["total"]))
    if len(metrics["priority"]) > 0:
        for p_dict in metrics["priority"]:
            if isinstance(p_dict, dict):
                priorities_found.update(p_dict.keys())

PRIORITY_MAX = collections.defaultdict(float)
for p in priorities_found:
    max_p = 1.0
    for metrics in data_dict.values():
        if len(metrics["priority"]) > 0:
            for p_dict in metrics["priority"]:
                 if isinstance(p_dict, dict) and p in p_dict:
                     max_p = max(max_p, p_dict[p])
    PRIORITY_MAX[p] = max_p + (max_p * 0.01)

MAX_TOTAL = MAX_TOTAL if MAX_TOTAL > 0 else 1.0

sorted_p = sorted([int(p) for p in priorities_found]) if priorities_found else [1, 2, 3, 4, 7, 10]

# ----------------------------------------
# 1. GENERATE FINAL TABLE METRICS
# ----------------------------------------
print("\n" + "="*80)
print("TABLE III: PERFORMANCE BENCHMARKING AT 30,000 EPISODES")
print("="*80)

# We print raw smoothed values at exactly 30,000 episodes (last index)
header_p_str = " | ".join([f"P{p} (%)".ljust(6) for p in sorted_p])
header = f"| {'Algorithm':<30} | {'Reward':<8} | {header_p_str} | {'Total (%)':<9} |"
print(header)
sep_p_str = "|".join(["-"*8 for _ in sorted_p])
print("|" + "-"*32 + "|" + "-"*10 + "|" + sep_p_str + "|" + "-"*11 + "|")

for run in runs:
    lbl = run["label"]
    if lbl not in data_dict: continue
    
    metrics = data_dict[lbl]
    
    # Get last smoothed reward
    smooth_rew = smooth(metrics["reward"])
    final_rew = smooth_rew[-1] if len(smooth_rew) > 0 else 0.0
    
    # Get last smoothed total
    smooth_tot = smooth([v / MAX_TOTAL for v in metrics["total"]])
    final_tot = (smooth_tot[-1] * 100) if len(smooth_tot) > 0 else 0.0
    
    p_vals = {}
    for p in sorted_p:
        if len(metrics["priority"]) > 0:
            raw_seq = [d.get(p, 0.0) if isinstance(d, dict) else 0.0 for d in metrics["priority"]]
            ratio_seq = [(v / PRIORITY_MAX.get(p, 1.0)) for v in raw_seq]
            smooth_p = smooth(ratio_seq)
            p_vals[p] = (min(1.0, smooth_p[-1]) * 100) if len(smooth_p) > 0 else 0.0
        else:
            p_vals[p] = 0.0
            
    p_row_str = " | ".join([f"{p_vals[p]:>6.1f}" for p in sorted_p])
    row = f"| {lbl:<30} | {final_rew:>8.1f} | {p_row_str} | {final_tot:>9.1f} |"
    print(row)
print("="*80 + "\n")

# Generate table as PDF
try:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')
    
    col_labels = ['Algorithm', 'Reward'] + [f'P{p} (%)' for p in sorted_p] + ['Total (%)']
    cell_text = []
    for run in runs:
        lbl = run["label"]
        if lbl not in data_dict: continue
        metrics = data_dict[lbl]
        smooth_rew = smooth(metrics["reward"])
        final_rew = smooth_rew[-1] if len(smooth_rew) > 0 else 0.0
        smooth_tot = smooth([v / MAX_TOTAL for v in metrics["total"]])
        final_tot = (smooth_tot[-1] * 100) if len(smooth_tot) > 0 else 0.0
        p_vals = {}
        for p in sorted_p:
            if len(metrics["priority"]) > 0:
                raw_seq = [d.get(p, 0.0) if isinstance(d, dict) else 0.0 for d in metrics["priority"]]
                ratio_seq = [(v / PRIORITY_MAX.get(p, 1.0)) for v in raw_seq]
                smooth_p = smooth(ratio_seq)
                p_vals[p] = (min(1.0, smooth_p[-1]) * 100) if len(smooth_p) > 0 else 0.0
            else:
                p_vals[p] = 0.0
        row_data = [lbl, f"{final_rew:.1f}"] + [f"{p_vals[p]:.1f}" for p in sorted_p] + [f"{final_tot:.1f}"]
        cell_text.append(row_data)

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    table_pdf_path = os.path.join(save_dir, "Performance_Benchmarking_Table.pdf")
    plt.savefig(table_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {table_pdf_path}")
except Exception as e:
    print(f"Error generating table PDF: {e}")

# ----------------------------------------
# 2. GENERATE PDF PLOTS (LOG ONLY)
# ----------------------------------------

def plot_single(filename, y_data_key, y_label, title):
    plt.figure(figsize=(10, 6))
    
    has_data = False
    for label, metrics in data_dict.items():
        d = metrics.get(y_data_key, [])
        if len(d) > 0:
            y = np.array(d)
            if y_data_key == "total":
                y = y / MAX_TOTAL
            
            y_smoothed = smooth(y.tolist(), box_pts=12)
            x_axis = [i * EVAL_CYCLE for i in range(len(d))]
            
            min_len = min(len(x_axis), len(y_smoothed))
            x_plot = x_axis[:min_len]
            y_plot = y_smoothed[:min_len]
            
            x_plot_f = [x for x in x_plot if x > 0]
            y_plot_f = [y for x, y in zip(x_plot, y_plot) if x > 0]
            
            # Evenly distribute markers visually in log space
            num_markers = 15
            if len(x_plot_f) > num_markers:
                log_min = np.log10(x_plot_f[0])
                log_max = np.log10(x_plot_f[-1])
                target_log_x = np.linspace(log_min, log_max, num_markers)
                target_x = 10 ** target_log_x
                
                marker_indices = []
                x_arr = np.array(x_plot_f)
                for tx in target_x:
                    idx = int(np.abs(x_arr - tx).argmin())
                    if idx not in marker_indices:
                        marker_indices.append(idx)
            else:
                marker_indices = list(range(len(x_plot_f)))
            
            plt.plot(x_plot_f, y_plot_f, label=label, color=metrics["run_cfg"]["c"], 
                     linestyle=metrics["run_cfg"]["ls"], linewidth=3.0,
                     marker=metrics["run_cfg"]["marker"], markersize=9, markevery=marker_indices)
            has_data = True
            
    if has_data:
        plt.title(f"{title}", fontsize=22)
        plt.xlabel("Episodes [Log Scale]", fontsize=20)
        plt.ylabel(y_label, fontsize=20)
        
        plt.xscale('log')
        plt.xlim(10, 30000)
        
        plt.grid(True, which='major', linestyle='-', alpha=0.7)
        plt.grid(True, which='minor', linestyle=':', alpha=0.5)
        
        plt.tick_params(axis='both', labelsize=18)
        plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
        plt.tight_layout()
        
        save_path_pdf = os.path.join(save_dir, f"{filename}.pdf")
        plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
        print(f"Saved: {save_path_pdf}")
        
    plt.close()

# Plot 1 & 2
plot_single("Comparison_of_Data_Collection_Ratios", "total", "Total Data Collection Ratio", "Comparison of Data Collection Ratios")
plot_single("Comparison_of_Total_Rewards", "reward", "Total Reward", "Comparison of Total Rewards\n(With Proposed Algorithm's Reward Formula)")

# Plot 3: Combined Priorities
cols = 3
num_p = max(1, len(sorted_p))
rows = (num_p + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
axes = axes.flatten()

for i, p in enumerate(sorted_p):
    ax = axes[i]
    for label, metrics in data_dict.items():
        if len(metrics["priority"]) > 0:
            raw_seq = [d.get(p, 0.0) if isinstance(d, dict) else 0.0 for d in metrics["priority"]]
            ratio_seq = [(v / PRIORITY_MAX.get(p, 1.0)) for v in raw_seq]
            
            y_smoothed = smooth(ratio_seq, box_pts=12)
            y_smoothed = [min(1.0, val) for val in y_smoothed]
            
            run_cfg = metrics["run_cfg"]
            x_axis = [i * EVAL_CYCLE for i in range(len(metrics["priority"]))]
            
            min_len = min(len(x_axis), len(y_smoothed))
            x_plot = x_axis[:min_len]
            y_plot = y_smoothed[:min_len]
            
            x_plot_f = [x for x in x_plot if x > 0]
            y_plot_f = [y for x, y in zip(x_plot, y_plot) if x > 0]
            
            # Evenly distribute markers visually in log space
            num_markers = 15
            if len(x_plot_f) > num_markers:
                log_min = np.log10(x_plot_f[0])
                log_max = np.log10(x_plot_f[-1])
                target_log_x = np.linspace(log_min, log_max, num_markers)
                target_x = 10 ** target_log_x
                
                marker_indices = []
                x_arr = np.array(x_plot_f)
                for tx in target_x:
                    idx = int(np.abs(x_arr - tx).argmin())
                    if idx not in marker_indices:
                        marker_indices.append(idx)
            else:
                marker_indices = list(range(len(x_plot_f)))
            
            ax.plot(x_plot_f, y_plot_f, label=label, color=run_cfg["c"], 
                    linestyle=run_cfg["ls"], linewidth=3.0,
                    marker=run_cfg["marker"], markersize=9, markevery=marker_indices)
            
    ax.set_title(f"Priority {p}", fontsize=22)
    ax.set_xlabel("Episode [Log Scale]", fontsize=20)
    ax.set_ylabel("Data collection ratio", fontsize=20)
    
    ax.set_xscale('log')
    ax.set_xlim(10, 30000)
    ax.set_ylim(0.0, 1.05)
    
    ax.grid(True, which='major', linestyle='-', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.5)
    ax.tick_params(axis='both', labelsize=18)

for j in range(len(sorted_p), len(axes)):
    fig.delaxes(axes[j])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=18, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=True)
plt.tight_layout(pad=4.0, rect=[0, 0.08, 1, 1])

combined_filename_pdf = os.path.join(save_dir, "All_Priority_Levels_Comparison.pdf")
plt.savefig(combined_filename_pdf, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {combined_filename_pdf}")

print("\nProcess finished successfully!")
