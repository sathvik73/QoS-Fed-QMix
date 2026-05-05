
import subprocess
import time
import sys
import os

def run_experiments():
    print("Starting Parallel Experiments (500 Episodes)...")
    #python main.py --map=RDM --federated=True --model=True --alg=qmix --n_agents=3 --tag=exact_paper_replication --total_episodes=3000 --anneal_steps=2000 --model_learning_period=1000

    # Baseline command (Removed as per user request)
    # cmd1 = [sys.executable, "main.py", "--total_episodes", "3000", "--anneal_steps", "2000", "--model_learning_period", "1000", "--alg", "qmix", "--map", "RDM", "--tag", "_baseline_varf", "--federated","True","--n_agents","3","--model","True"]
    
    # Proposed Command: Priority Aware + Deadline + Load Balanced
    # Forward any command line arguments (e.g., --device cuda) to the internal command
    extra_args = sys.argv[1:]
    cmd2 = [sys.executable, "main.py", "--total_episodes", "3000", "--anneal_steps", "2000", "--model_learning_period", "1000", "--alg", "qmix", "--map", "RDM", "--tag", "_priority_final", "--priority", "--federated","True","--n_agents","3","--model","True"] + extra_args


    # p1 = subprocess.Popen(cmd1)
    p2 = subprocess.Popen(cmd2)
    
    print(f"Experiment started with PID: {p2.pid}")
    
    exit_codes = [p2.wait()]
    
    if any(code != 0 for code in exit_codes):
        print("Experiment failed.")
    else:
        print("Experiment completed successfully.")
        
    # print("Generating Plots...")
    # plot_cmd = [sys.executable, "plot_comparisons.py", 
    #             "--baseline", "result/qmix/RDM_baseline_varf", 
    #             "--proposed", "result/qmix/RDM_priority_varf", 
    #             "--save_dir", "result/comparison_varf"]
    # 
    # subprocess.run(plot_cmd, check=True)
    print("Done.")


if __name__ == "__main__":
    run_experiments()
