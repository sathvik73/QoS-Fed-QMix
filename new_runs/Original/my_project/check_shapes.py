import os
import sys
import numpy as np

base_dir = r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting"

paths = {
    "proposed": r"proposed_run\result\qmix\RDM_proposedf",
    "modelqmix": r"modelqmix_run\result\qmix\RDM_model_qmix",
    "qmix": r"qmix_run\result\qmix\RDM_qmix",
    "original": r"original_run\result\qmix\RDM_original"
}

with open("shapes.txt", "w") as f:
    for name, rel_path in paths.items():
        f.write(f"\n--- {name} ---\n")
        folder = os.path.join(base_dir, rel_path)
        if not os.path.exists(folder):
            f.write(f"Folder not found: {folder}\n")
            continue
        
        files_to_check = ["episode_data_.npy", "episode_rewards_.npy", "episode_priority_data_.npy",
                          "global_data.npy", "global_rewards.npy", "episode_priority_data_0.npy"]
        
        for file in files_to_check:
            full_path = os.path.join(folder, file)
            if os.path.exists(full_path):
                arr = np.load(full_path, allow_pickle=True)
                if hasattr(arr, "shape"):
                    f.write(f"{file}: shape = {arr.shape}\n")
                else:
                    f.write(f"{file}: type = {type(arr)}, len = {len(arr)}\n")
