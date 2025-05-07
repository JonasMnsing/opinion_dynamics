import numpy as np
import os
import sys
sys.path.append("src/")
from voter_model import *

# Parameters
N_agents        = 50
N_iter          = 1000
repetitions     = 100
p_con           = 0.1
p_noise_values  = np.linspace(0.0, 0.2, 10)

# Create output directory
os.makedirs("data/voter/noise", exist_ok=True)

# Simulation loop
for idx, p_noise in enumerate(p_noise_values):
    all_avg_opinions    = np.zeros((repetitions, N_iter))
    all_int_densities   = np.zeros((repetitions, N_iter))

    for rep in range(repetitions):
        opinions    = initialize_opinions(N_agents)
        con_matrix  = connected_erdos_renyi(N_agents, p_con)

        avg_opinion_t = np.zeros(N_iter)
        int_density_t = np.zeros(N_iter)

        for t in range(N_iter):
            # Modified voter interaction with noise
            opinions            = voter_interaction(opinions, con_matrix, p_noise=p_noise)
            avg_opinion_t[t]    = average_opinion(opinions)
            int_density_t[t]    = interface_density(opinions, con_matrix)
    
        all_avg_opinions[rep]   = avg_opinion_t
        all_int_densities[rep]  = int_density_t

        print(f"noise={p_noise:.3f}, run {rep+1}/{repetitions} completed.")

    # Save for this noise level
    np.save(f"data/voter/noise/avg_opinion_noise_{idx}.npy", all_avg_opinions)
    np.save(f"data/voter/noise/int_density_noise_{idx}.npy", all_int_densities)