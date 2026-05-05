
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def plot_comparisons(baseline_dir, proposed_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load data
    eval_cycle = 50 # Default or from args if added
    try:
        # But runner.py saves locally per evaluation cycle. 
        # Actually runner.py saves 'episode_data_{num}.npy' 
        # and also 'training_episode_data_{}.npy' at the end.
        
        # We should probably look for 'training_episode_data_{}.npy' or 'episode_data_{}.npy' 
        # Let's look for the concatenated files if they exist, or just the evaluation logs
        
        # Let's try to find the latest evaluation log file
        def load_latest_log(directory, prefix):
            files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.npy')]
            if not files:
                return None
            # Sort by number? The format is prefix_{num}.npy. If num is empty string for the first one... 
            # runner.py: np.save(self.save_path + '/episode_data_{}'.format(str(num)), self.episode_data)
            # If num is loop index, it's an integer. 
            
            # Actually, runner.py appends to self.episode_data list and saves the WHOLE list every time plt() is called.
            # So we just need the file with the largest number, or simply the one with the latest modification time, 
            # or even better, just any non-empty one since they contain the full history so far? 
            # Wait, plt(num) saves self.episode_data, which is a list that grows. 
            # So the last file saved contains all data.
            
            # Let's verify the file naming in runner.py
            # self.plt(num='') calls:
            # np.save(self.save_path + '/episode_data_{}'.format(str(num)), self.episode_data)
            # resulting in 'episode_data_.npy'
            
            # self.plt(i) calls:
            # np.save(self.save_path + '/episode_data_{}'.format(str(i)), self.episode_data)
            
            # So we should look for the file with the largest integer index, or falls back to '_'
            
            max_idx = -1
            target_file = None
            
            for f in files:
                parts = f.replace(prefix, '').replace('.npy', '')
                if parts == '_':
                    idx = -1
                elif parts.startswith('_'):
                     # _0, _1 etc
                     try:
                        idx = int(parts[1:])
                     except:
                        idx = -1
                else:
                    idx = -1
                
                if idx > max_idx:
                    max_idx = idx
                    target_file = f
            
            if target_file:
                return np.load(os.path.join(directory, target_file), allow_pickle=True)
            else:
                 # Fallback to 'episode_data_.npy' or other variations if exists
                 candidates = [
                     prefix + '_.npy',
                     prefix + '.npy'
                 ]
                 for cand in candidates:
                     if os.path.exists(os.path.join(directory, cand)):
                         return np.load(os.path.join(directory, cand), allow_pickle=True)
                 return None


        if baseline_dir:
            base_data = load_latest_log(baseline_dir, 'episode_data')
            base_rewards = load_latest_log(baseline_dir, 'episode_rewards')
            base_priority = load_latest_log(baseline_dir, 'episode_priority_data')
        else:
            base_data = None
            base_rewards = None
            base_priority = None

        prop_data = load_latest_log(proposed_dir, 'episode_data')
        prop_rewards = load_latest_log(proposed_dir, 'episode_rewards')
        prop_priority = load_latest_log(proposed_dir, 'episode_priority_data')
        
        if prop_data is None:
             print("Could not load proposed data.")
             return

        # Plot Total Data Collection
        plt.figure(figsize=(10, 6))
        
        # Generate x-axis
        if base_data is not None:
             x_base = np.arange(len(base_data)) * eval_cycle
             plt.plot(x_base, base_data, label='Baseline')
        
        x_prop = np.arange(len(prop_data)) * eval_cycle
        plt.plot(x_prop, prop_data, label='Proposed')
        plt.xlabel('Episodes')
        plt.ylabel('Total Collected Data')
        plt.title('Total Data Collection Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'comparison_data_total.png'))
        plt.close()
        
        # Plot Total Rewards
        plt.figure(figsize=(10, 6))
        
        # Plot Total Rewards
        plt.figure(figsize=(10, 6))
        
        if base_rewards is not None:
            x_base_rew = np.arange(len(base_rewards)) * eval_cycle
            plt.plot(x_base_rew, base_rewards, label='Baseline')
            
        x_prop_rew = np.arange(len(prop_rewards)) * eval_cycle
        plt.plot(x_prop_rew, prop_rewards, label='Proposed')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Total Reward Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'comparison_rewards_total.png'))
        plt.close()

        base_priority = load_latest_log(baseline_dir, 'episode_priority_data')
        prop_priority = load_latest_log(proposed_dir, 'episode_priority_data')
        
        if base_priority is None and prop_priority is None:
             print("No priority breakdown data found. Skipping detailed plots.")
        else:
            # Get all unique priorities found
            priorities = set()
            if prop_priority is not None:
                for d in prop_priority:
                    priorities.update(d.keys())
            if base_priority is not None:
                for d in base_priority:
                     priorities.update(d.keys())
            
            sorted_priorities = sorted(list(priorities))
            
            for p in sorted_priorities:
                # Extract series
                if base_priority is not None:
                    base_p_data = [d.get(p, 0) for d in base_priority]
                    base_p_reward = [val * p for val in base_p_data]
                else:
                    base_p_data = None
                    base_p_reward = None
                
                if prop_priority is not None:
                    prop_p_data = [d.get(p, 0) for d in prop_priority]
                    prop_p_reward = [val * p for val in prop_p_data]
                else:
                    prop_p_data = None
                    prop_p_reward = None

                # Plot Data P{p}
                plt.figure(figsize=(10, 6))
                if base_p_data is not None:
                    plt.plot(np.arange(len(base_p_data)) * eval_cycle, base_p_data, label='Baseline')
                if prop_p_data is not None:
                    plt.plot(np.arange(len(prop_p_data)) * eval_cycle, prop_p_data, label='Proposed')
                plt.xlabel('Episodes')
                plt.ylabel(f'Collected Data (Priority {p})')
                plt.title(f'Data Collection - Priority {p}')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(save_dir, f'comparison_data_p{p}.png'))
                plt.close()
                
                # Plot Reward P{p}
                plt.figure(figsize=(10, 6))
                if base_p_reward is not None:
                    plt.plot(np.arange(len(base_p_reward)) * eval_cycle, base_p_reward, label='Baseline')
                if prop_p_reward is not None:
                    plt.plot(np.arange(len(prop_p_reward)) * eval_cycle, prop_p_reward, label='Proposed')
                plt.xlabel('Episodes')
                plt.ylabel(f'Reward (Priority {p})')
                plt.title(f'Reward - Priority {p}')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(save_dir, f'comparison_rewards_p{p}.png'))
                plt.close()

        print(f"Comparison plots saved to {save_dir}")
        
    except Exception as e:
        print(f"Error plotting comparisons: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, default=None, help='path to baseline results (optional)')
    parser.add_argument('--proposed', type=str, required=True, help='path to proposed results')
    parser.add_argument('--save_dir', type=str, default='result/comparison_var', help='path to save comparison plots')


    args = parser.parse_args()
    
    plot_comparisons(args.baseline, args.proposed, args.save_dir)
