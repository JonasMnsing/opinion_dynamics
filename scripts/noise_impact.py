import os
import numpy as np
import sys
sys.path.append("src/")
from opinion_dynamics import OpinionDynamicsModel, connected_erdos_renyi
from tqdm import tqdm

# --- Parameter sweep configuration ---
p_noise_list    = np.linspace(0, 1.0, 101) # Noise levels from 0 to 1
p_conn          = 1.0
n_runs          = 50
N               = 80
max_steps       = 1000
n_processes     = 10
path            = "data/social_impact/"
pressure        = np.random.uniform(-1,1, size=N)
support         = np.random.uniform(-1,1, size=N)

# Ensure output directory exists
os.makedirs(path, exist_ok=True)

results = []
params  = []

for p_noise in tqdm(p_noise_list):
    model = OpinionDynamicsModel(N_agents=N, graph_fn=connected_erdos_renyi,
                                    graph_kwargs={'p': p_conn}, p_noise=p_noise, pressure=pressure, support=support)
    stats = model.ensemble_stats(n_runs=n_runs, max_steps=max_steps,
                                    stop_on_consensus=False, n_processes=n_processes)
    
    results.append(stats['m_trajs'])
    params.append([N, n_runs, p_conn, p_noise])

np.save(f"{path}er_noise.npy", np.array(results))
np.save(f"{path}er_noise_params.npy", np.array(params))