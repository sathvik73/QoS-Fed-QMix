import numpy as np

arr1 = np.load(r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting\proposed_run\result\qmix\RDM_proposedf\global_data.npy", allow_pickle=True)
arr2 = np.load(r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting\modelqmix_run\result\qmix\RDM_model_qmix\episode_data_.npy", allow_pickle=True)

print("Proposed global data info:")
print("- Max:", np.max(arr1))
print("- Min:", np.min(arr1))

print("ModelQMIX episode data info:")
print("- Max:", np.max(arr2))
print("- Min:", np.min(arr2))
