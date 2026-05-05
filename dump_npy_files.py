import os

base_dir = r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting"
runs = ["modelqmix_run", "original_run", "proposed_run", "qmix_run"]

with open("npy_files.txt", "w") as f:
    for run in runs:
        folder_path = os.path.join(base_dir, run)
        if not os.path.exists(folder_path):
            f.write(f"Directory not found: {folder_path}\n")
            continue
        
        f.write(f"\n--- {run} ---\n")
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.npy'):
                    rel_path = os.path.relpath(os.path.join(root, file), folder_path)
                    f.write(rel_path + "\n")
