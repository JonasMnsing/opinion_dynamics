import os
import numpy as np
import sys
sys.path.append("src/")
from opinion_dynamics import OpinionDynamicsModel, scale_free_graph
from tqdm import tqdm

# --- Parameter sweep configuration ---
path            = "data/social_impact/"
p_scale_free    = np.arange(1,11)
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

for m in tqdm(p_scale_free):
    model = OpinionDynamicsModel(N_agents=N, graph_fn=scale_free_graph,
                                    graph_kwargs={'m': m}, p_noise=0.1, beta=beta)
    stats = model.ensemble_stats(n_runs=n_runs, max_steps=max_steps,
                                    stop_on_consensus=False, n_processes=n_processes)
    
    m_traj.append(stats['m_trajs'])
    o_traj.append(stats['o_trajs'])
    params.append([N, n_runs, m])

np.save(f"{path}scale_free.npy", np.array(m_traj))
np.save(f"{path}scale_free_opinion.npy", np.array(o_traj))
np.save(f"{path}scale_free_params.npy", np.array(params))
