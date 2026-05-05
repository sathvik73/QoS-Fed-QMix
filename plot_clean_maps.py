import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from Libs.Utils.Plots import plot_city_top_view
from config.RDM_define import params as rdm_params
from config.RBM_define import params as rbm_params
import os

def generate_clean_map(params, map_name):
    # Setup
    ax = plot_city_top_view(params['city'], 10, figsize=(8,6), fontsize=12)
    device_position = params['device_position'].copy()
    
    unknown_idx = params['unknown_device_idx']
    known_idx = params['known_device_idx']
    
    unknown_device_position = device_position[unknown_idx]
    unknown_device_position_colors = np.array(params['color'])[unknown_idx]
    
    anchor_nodes = device_position[known_idx]
    anchor_nodes_colors = np.array(params['color'])[known_idx]

    unknown_device_position = np.resize(unknown_device_position, (len(unknown_device_position), 3))
    anchor_nodes = np.resize(anchor_nodes, (len(anchor_nodes), 3))

    n_agents = len(params['start_pose'])
    uav_start_pose = np.resize(params['start_pose'][:n_agents], (n_agents, 3))
    uav_terminal_pose = np.resize(params['end_pose'][:n_agents], (n_agents, 3))

    # Plot features
    plt.scatter(unknown_device_position[:, 0], unknown_device_position[:, 1], marker='*', s=100,
                c=unknown_device_position_colors, label='Unknown Device Position', zorder=10)
    plt.scatter(anchor_nodes[:, 0], anchor_nodes[:, 1], marker='^', s=100, c=anchor_nodes_colors, 
                label='Anchor Device Position', zorder=10)

    plt.scatter(uav_start_pose[:, 0], uav_start_pose[:, 1], marker='H', s=150, c='lightgray',
                label='UAV Start Zone', zorder=5)
    plt.scatter(uav_terminal_pose[:, 0], uav_terminal_pose[:, 1], marker='H', s=150, c='lightblue',
                label='UAV Terminal Zone', zorder=5)

    # Adjust legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=2, fontsize=11, columnspacing=0.5, handlelength=1)

    # Save
    out_dir = "result/clean_maps"
    os.makedirs(out_dir, exist_ok=True)
    
    plt.title(f"{map_name} Clean Map")
    plt.savefig(f"{out_dir}/{map_name}_clean.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{out_dir}/{map_name}_clean.png", format='png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    print(f"Generated {map_name} clean map in {out_dir}/")

if __name__ == '__main__':
    generate_clean_map(rdm_params, "RDM")
    generate_clean_map(rbm_params, "RBM")
    print("Done!")
