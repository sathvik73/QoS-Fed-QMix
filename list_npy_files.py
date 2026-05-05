import os
import sys

base_dir = r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting"
runs = ["modelqmix_run", "original_run", "proposed_run", "qmix_run"]

print("Finding all .npy files in the 4 run directories...")
for run in runs:
    folder_path = os.path.join(base_dir, run)
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        continue
    
    print(f"\n--- {run} ---")
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.endswith('.npy'):
                rel_path = os.path.relpath(os.path.join(root, f), folder_path)
                print(rel_path)
