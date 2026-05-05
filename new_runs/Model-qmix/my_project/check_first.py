import numpy as np

# proposed global data
arr1 = np.load(r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting\proposed_run\result\qmix\RDM_proposedf\global_data.npy", allow_pickle=True)
print("Proposed (0, 50, 100):", arr1[:3])

# modelqmix episode data
arr2 = np.load(r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting\modelqmix_run\result\qmix\RDM_model_qmix\episode_data_.npy", allow_pickle=True)
print("ModelQMIX (50, 100):", arr2[:2])
