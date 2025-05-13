import os
import numpy as np
import pandas as pd
import sys
sys.path.append("src/")
from voter_model import OpinionDynamicsModel, connected_erdos_renyi
from tqdm import tqdm

# --- Parameter sweep configuration ---
N_list          = [10, 20, 40, 80, 160, 320, 640]
p_conn_list     = [0.01, 0.02, 0.04, 0.08, 0.016]
p_noise_list    = [0.0, 0.01, 0.02, 0.04, 0.08]
n_runs          = 50
max_steps       = 10000
n_processes     = 10
path            = "data"

# Ensure output directory exists
os.makedirs(path, exist_ok=True)

# Collect results in a list of dicts
all_results = []

for N in tqdm(N_list):
    for p_conn in p_conn_list:
        for p_noise in p_noise_list:
            model = OpinionDynamicsModel(N_agents=N, graph_fn=connected_erdos_renyi,
                                         graph_kwargs={'p': p_conn}, p_noise=p_noise)
            stats = model.ensemble_stats(n_runs=n_runs, max_steps=max_steps,
                                         stop_on_consensus=True, n_processes=n_processes)
            
            frac_consensus = np.mean([t < max_steps for t in stats['times']])
            all_results.append({
                'N': N,
                'p_connection': p_conn,
                'p_noise': p_noise,
                'mean_consensus_time': stats['mean_time'],
                'fraction_consensus': frac_consensus
            })

# Convert to DataFrame
df = pd.DataFrame(all_results)

# Save raw results
df.to_csv('data/connected_erdos_renyi_results.csv', index=False)

# Pivot tables
pivot_time = df.pivot_table(
    index=['N', 'p_connection'],
    columns='p_noise',
    values='mean_consensus_time'
)
pivot_prob = df.pivot_table(
    index=['N', 'p_connection'],
    columns='p_noise',
    values='fraction_consensus'
)

# Save pivot tables
pivot_time.to_csv('data/connected_erdos_renyi_pivot_time.csv')
pivot_prob.to_csv('data/connected_erdos_renyi_pivot_prob.csv')

# Display summaries
print("Mean consensus time (by N, p_connection, p_noise):")
print(pivot_time)
print("\nConsensus probability (fraction reaching consensus):")
print(pivot_prob)