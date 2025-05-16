import os
import numpy as np
import sys
sys.path.append("src/")
from voter_model import OpinionDynamicsModel, connected_erdos_renyi
from tqdm import tqdm

# --- Parameter sweep configuration ---
p_conn_list     = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
p_noise_list    = [0.0, 0.01, 0.02, 0.04, 0.08]
n_runs          = 50
N               = 50
max_steps       = 1000
n_processes     = 10
path            = "data"

# Ensure output directory exists
os.makedirs(path, exist_ok=True)

results = []
params  = []

for p_conn in tqdm(p_conn_list):
    for p_noise in p_noise_list:
        model = OpinionDynamicsModel(N_agents=N, graph_fn=connected_erdos_renyi,
                                        graph_kwargs={'p': p_conn}, p_noise=p_noise)
        stats = model.ensemble_stats(n_runs=n_runs, max_steps=max_steps,
                                        stop_on_consensus=False, n_processes=n_processes)
        
        results.append(stats['m_trajs'])
        params.append([p_conn, p_noise])

np.save(f"data/voter/m_trajs.npy", np.array(results))
np.save(f"data/voter/params.npy", np.array(params))