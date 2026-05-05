
import os
import sys

print("Current CWD:", os.getcwd(), flush=True)
print("sys.path:", sys.path, flush=True)

try:
    print("Importing numpy...", flush=True)
    import numpy as np
    print("Numpy imported.", flush=True)
except Exception as e:
    print("Error importing numpy:", e, flush=True)

try:
    print("Importing Libs.Utils.utils...", flush=True)
    from Libs.Utils.utils import generate_uav_trajectory_from_points, sample_from_line, inpolygon
    print("Libs.Utils.utils imported.", flush=True)
except Exception as e:
    print("Error importing Libs.Utils.utils:", e, flush=True)

try:
    print("Importing config.RDM_define...", flush=True)
    from config.RDM_define import params
    print("config.RDM_define imported.", flush=True)
except Exception as e:
    print("Error importing config.RDM_define:", e, flush=True)

print("Debug script finished.", flush=True)
