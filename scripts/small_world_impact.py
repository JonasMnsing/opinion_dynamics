import os
import numpy as np
import sys
sys.path.append("src/")
from opinion_dynamics import OpinionDynamicsModel, small_world_graph
from tqdm import tqdm

# --- Parameter sweep configuration ---
path            = "data/social_impact/"
p_small_world   = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
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

for p in tqdm(p_small_world):
    model = OpinionDynamicsModel(N_agents=N, graph_fn=small_world_graph,
                                    graph_kwargs={'k': 4, 'beta': p}, p_noise=0.1, beta=beta)
    stats = model.ensemble_stats(n_runs=n_runs, max_steps=max_steps,
                                    stop_on_consensus=False, n_processes=n_processes)
    
    m_traj.append(stats['m_trajs'])
    o_traj.append(stats['o_trajs'])
    params.append([N, n_runs, p])

np.save(f"{path}small_world.npy", np.array(m_traj))
np.save(f"{path}small_world_opinion.npy", np.array(o_traj))
np.save(f"{path}small_world_params.npy", np.array(params))
