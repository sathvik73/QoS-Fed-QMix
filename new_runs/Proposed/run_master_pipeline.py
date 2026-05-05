import subprocess
import time
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Master Pipeline to run all 4 multi-UAV Data Harvesting models sequentially.")
    parser.add_argument("--total_episodes", type=int, default=30000, help="Total episodes to run per model (default: 30000). Set lower for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Global environment seed to guarantee exact identical digital twins across models.")
    args, extra_args = parser.parse_known_args()

    print("=" * 80)
    print(f"Starting Unified Digital Twin Master Pipeline ({args.total_episodes} Episodes)")
    print(f"Global Seed: {args.seed} (Guarantees identical buildings, IoT locations, and deadlines)")
    print("=" * 80)
    print()

    # Common parameters matching the original bat script structure, plus our enforced seed
    common_args = [
        "--total_episodes", str(args.total_episodes),
        "--anneal_steps", "2000",
        "--model_learning_period", "1000",
        "--alg", "qmix",
        "--map", "RDM",
        "--n_agents", "3",
        "--load_model", "True",
        "--evaluate_cycle", "1",
        "--seed", str(args.seed)
    ]

    # Model Configurations
    models = [
        {
            "name": "Proposed FedQMIX (Priority Aware + Deadline + Load Balanced)",
            "args": ["--model", "True", "--priority", "--federated", "True", "--tag", "_proposedf"]
        },
        {
            "name": "Original FedQMIX (Baseline)",
            "args": ["--model", "True", "--federated", "True", "--tag", "_original"]
        },
        {
            "name": "Model-Aided QMIX (Centralized QMIX with Channel Learning)",
            "args": ["--model", "True", "--federated", "False", "--tag", "_model_qmix"]
        },
        {
            "name": "Pure QMIX (Standard Centralized QMIX)",
            "args": ["--model", "False", "--federated", "False", "--tag", "_qmix"]
        }
    ]

    start_time = time.time()

    for i, model in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] Starting: {model['name']}")
        print("-" * 50)
        
        cmd = [sys.executable, "main.py"] + common_args + model["args"] + extra_args
        print(f"Executing: {' '.join(cmd)}")
        
        try:
            # Run sequentially
            process = subprocess.Popen(cmd)
            exit_code = process.wait()
            
            if exit_code != 0:
                print(f"WARNING: Model {model['name']} exited with non-zero code ({exit_code}).")
                # We optionally could halt here, but we'll try to push through in case of minor timeouts
        except KeyboardInterrupt:
            print("\nPipeline interrupted by user.")
            process.terminate()
            sys.exit(1)

    total_time = (time.time() - start_time) / 3600
    print(f"\nAll 4 simulations completed in {total_time:.2f} hours.")
    print("=" * 80)
    print("Triggering Final Plot Generation...")
    
    plot_cmd = [sys.executable, "generate_final_plots.py"]
    try:
        subprocess.run(plot_cmd, check=True)
        print("Plots generated successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Plot generation failed with error: {e}")

if __name__ == "__main__":
    main()
