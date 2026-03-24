import subprocess
import sys
import os

# Define the models to run sequentially to avoid overloading the CPU/RAM
experiments = [
    {
        "name": "Proposed (Priority-Aware Model-Aided FedQMIX)",
        "args": ["--model", "True", "--federated", "True", "--priority", "--tag", "_proposed"]
    },
    {
        "name": "Baseline FedQMIX (Model-Aided FedQMIX without Priority)",
        "args": ["--model", "True", "--federated", "True", "--tag", "_baseline_fedqmix"]
    },
    {
        "name": "Model-Aided QMIX (Centralized QMIX with Channel Learning)",
        "args": ["--model", "True", "--federated", "False", "--tag", "_model_qmix"]
    },
    {
        "name": "QMIX (Pure Centralized QMIX)",
        "args": ["--model", "False", "--federated", "False", "--tag", "_pure_qmix"]
    }
]

# Shared parameters for a quick test or full run
common_args = [
    "--total_episodes", "3000",   # Adjust to 3000 for full run, 100 for test
    "--anneal_steps", "2000",
    "--model_learning_period", "1000",
    "--alg", "qmix", 
    "--map", "RDM",
    "--n_agents", "3"
]

def run_experiments():
    print("=========================================================")
    print("Starting Comprehensive 4-Method Comparison (Sequential)")
    print("=========================================================\n")
    
    for exp in experiments:
        print(f"--> Starting: {exp['name']}")
        
        # Build the exact command list
        cmd = [sys.executable, "main.py"] + common_args + exp["args"]
        print(f"Command: {' '.join(cmd)}\n")
        
        # Run the experiment and wait for it to finish
        try:
            # We don't capture output here so the user can see progress in the terminal
            process = subprocess.run(cmd)
            if process.returncode != 0:
                print(f"!!! Error running {exp['name']}. Exit code: {process.returncode}")
                break
        except Exception as e:
            print(f"Exception occurred while running {exp['name']}: {e}")
            break
            
        print(f"\n--> Finished: {exp['name']}\n")
        print("---------------------------------------------------------")
        
    print("\nAll experiments completed.")

if __name__ == "__main__":
    run_experiments()
