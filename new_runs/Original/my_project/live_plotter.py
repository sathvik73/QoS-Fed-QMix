import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Define the models we are tracking
METHODS = [
    {"label": "Proposed", "dir": "result/qmix/RDM_proposed"},
    {"label": "Baseline FedQMIX", "dir": "result/qmix/RDM_baseline_fedqmix"},
    {"label": "Model-Aided QMIX", "dir": "result/qmix/RDM_model_qmix"},
    {"label": "QMIX", "dir": "result/qmix/RDM_pure_qmix"}
]

# Total data in the environment (sum of all device data)
# Based on [20000] * 10 devices = 200000. Hardcoded for ratio plots.
TOTAL_AVAILABLE_DATA = 200000.0

def load_data(dir_path, filename):
    filepath = os.path.join(dir_path, filename)
    if os.path.exists(filepath):
        try:
            return np.load(filepath, allow_pickle=True)
        except Exception:
            return None
    return None

def plot_consolidated_figure(fig_title, filename, is_ratio=False, is_log=False):
    # Create an 11-subplot figure (3x4 grid)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(fig_title, fontsize=20, fontweight='bold')
    axes = axes.flatten()
    
    # Hide the 12th empty subplot
    axes[11].axis('off')

    # We plot: Total (index 0), Priority 1-10 (indices 1-10)
    titles = ["Total Data"] + [f"Priority {p} Data" for p in range(1, 11)]
    
    # Set titles for all 11 active axes beforehand
    for idx, title in enumerate(titles):
        axes[idx].set_title(title)
        axes[idx].set_xlabel("Episodes")
        axes[idx].set_ylabel("Data Ratio" if is_ratio else "Data Collected")
        if is_log:
            axes[idx].set_xscale('log')
        if is_ratio:
            axes[idx].set_ylim(-0.05, 1.05)
            
    # Keep track of x_axis for later formatting if needed
    max_x = 0

    for method in METHODS:
        # Default Federated files
        priority_data_path = "episode_priority_data_0.npy"
        total_data_path = "global_data.npy"
        
        p_data = load_data(method["dir"], priority_data_path)
        t_data = load_data(method["dir"], total_data_path)
        
        # Fallback for Centralized files (no '0' agent index, no global merge)
        if p_data is None:
            p_data = load_data(method["dir"], "episode_priority_data_.npy")
        if t_data is None:
            t_data = load_data(method["dir"], "episode_data_.npy")
            
        if p_data is None or t_data is None:
            continue
            
        # Optional: Truncate to maximum 2000 episodes corresponding to the new tests
        # evaluate_cycle = 50, so 2000/50 + 1 (for step 0) = 41 data points.
        max_points = 41
        t_data = t_data[:max_points]
        p_data = p_data[:max_points]
            
        x_axis = np.arange(1, len(t_data) + 1) * 50
        if len(x_axis) > max_x: max_x = len(x_axis)
        
        # Subplot 0: Total Data
        y_total = t_data
        if is_ratio:
            # The environment has roughly 200,000 total data initially (10 devices * 20000)
            y_total = y_total / TOTAL_AVAILABLE_DATA
            
        axes[0].plot(x_axis, y_total, label=method["label"], linewidth=2)
        
        # Subplots 1-10: Priority Data
        for p in range(1, 11):
            y_p = []
            for ep_dict in p_data:
                # ep_dict might be a true dict or collections.defaultdict
                val = ep_dict.get(p, 0.0) 
                y_p.append(val)
                
            y_p = np.array(y_p)
            
            if is_ratio:
                # Assuming priority i has roughly (Total / 10) data. This is an approximation
                # since priorities are assigned randomly per run. Using Total Available for ratio
                # ensures it's mathematically sound, representing the % of the WHOLE pie.
                y_p = y_p / TOTAL_AVAILABLE_DATA
                
            axes[p].plot(x_axis, y_p, label=method["label"], linewidth=2)

    # Add legend to the first plot only to save space
    axes[0].legend(loc='lower right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    print("Live Plotter Started. Monitoring 'result/qmix/' directories...")
    print("Will generate 4 consolidated images every 60 seconds.")
    print("Press Ctrl+C to stop.")
    
    while True:
        try:
            # 1. Normal Scale, Absolute Data
            plot_consolidated_figure(
                fig_title="Collected Data Breakdown (Normal Scale)", 
                filename="Live_Collected_Data_Normal.png", 
                is_ratio=False, 
                is_log=False
            )
            
            # 2. Normal Scale, Ratio
            plot_consolidated_figure(
                fig_title="Data Collection Ratio Breakdown (Normal Scale)", 
                filename="Live_Data_Ratio_Normal.png", 
                is_ratio=True, 
                is_log=False
            )
            
            # 3. Log Scale, Absolute Data
            plot_consolidated_figure(
                fig_title="Collected Data Breakdown (Log Scale)", 
                filename="Live_Collected_Data_Log.png", 
                is_ratio=False, 
                is_log=True
            )
            
            # 4. Log Scale, Ratio
            plot_consolidated_figure(
                fig_title="Data Collection Ratio Breakdown (Log Scale)", 
                filename="Live_Data_Ratio_Log.png", 
                is_ratio=True, 
                is_log=True
            )
            
            print(f"[{time.strftime('%X')}] Plots updated.")
            time.sleep(60) # Wait 1 minute before polling again
            
        except Exception as e:
            print(f"Error during plotting: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
