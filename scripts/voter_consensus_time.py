import numpy as np
import os
import sys
from tqdm import tqdm
sys.path.append("src/")
from voter_model import *

# Parameters
N_agents        = 50
N_iter          = 100000
repetitions     = 100
p_con_values    = np.linspace(0.05, 1.0, 39)

# Store consensus times
consensus_times = np.zeros((len(p_con_values), repetitions))

# Create output directory
os.makedirs("data/voter/p_con", exist_ok=True)

# Simulation loop
for idx, p_con in enumerate(p_con_values):
    for rep in range(repetitions):
        opinions    = initialize_opinions(N_agents)
        con_matrix  = connected_erdos_renyi(N_agents, p_con)

        for t in range(N_iter):
            opinions = voter_interaction(opinions, con_matrix)
            if np.all(opinions == opinions[0]):  # Check for consensus
                consensus_times[idx, rep] = t
                break
        else:
            # If consensus not reached within max_iter
            consensus_times[idx, rep] = np.nan

        print(f"p={p_con:.3f}, run {rep+1}/{repetitions}, time to consensus: {consensus_times[idx, rep]}")

# Save results
np.save("data/voter/p_con/consensus_times.npy", consensus_times)
print("All simulations completed and results saved.")
