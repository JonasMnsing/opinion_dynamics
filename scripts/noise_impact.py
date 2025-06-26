import os
import numpy as np
import sys
sys.path.append("src/")
from opinion_dynamics import OpinionDynamicsModel, connected_erdos_renyi
from tqdm import tqdm

# --- Parameter sweep configuration ---
path            = "data/social_impact/"
p_noise_list    = np.linspace(0, 0.2, 21)
p_conn          = 0.1
n_runs          = 50
N               = 80
max_steps       = 1000
n_processes     = 10
beta            = 1.0

# Ensure output directory exists
os.makedirs(path, exist_ok=True)

m_traj  = []
o_traj  = []
params  = []

for p_noise in tqdm(p_noise_list):
    model = OpinionDynamicsModel(N_agents=N, graph_fn=connected_erdos_renyi,
                                    graph_kwargs={'p': p_conn}, p_noise=p_noise)
    stats = model.ensemble_stats(n_runs=n_runs, max_steps=max_steps,
                                    stop_on_consensus=False, n_processes=n_processes)
    
    m_traj.append(stats['m_trajs'])
    o_traj.append(stats['o_trajs'])
    params.append([N, n_runs, p_conn, p_noise])

np.save(f"{path}erdos_noise.npy", np.array(m_traj))
np.save(f"{path}erdos_noise_opinion.npy", np.array(o_traj))
np.save(f"{path}erdos_noise_params.npy", np.array(params))