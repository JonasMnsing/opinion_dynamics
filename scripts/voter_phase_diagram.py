import numpy as np
import os
import sys
sys.path.append("src/")
from voter_model import *

# Parameters
N_agents        = 50
N_iter          = 5000
repetitions     = 50

p_con_values    = np.linspace(0.05, 0.2, 17)
p_noise_values  = np.linspace(0.0, 0.1, 17)

# Output folder
os.makedirs("data/voter/phase", exist_ok=True)

# Data array to hold final average opinions
# Shape: (len(p_con), len(p_noise), repetitions)
final_avg_opinions = np.zeros((len(p_con_values), len(p_noise_values), repetitions))

for i, p_con in enumerate(p_con_values):
    for j, p_noise in enumerate(p_noise_values):
        for rep in range(repetitions):
            opinions   = initialize_opinions(N_agents)
            con_matrix = connected_erdos_renyi(N_agents, p_con)

            for _ in range(N_iter):
                opinions = voter_interaction(opinions, con_matrix, p_noise=p_noise)

            final_avg_opinions[i, j, rep] = average_opinion(opinions)

            print(f"p_con={p_con:.5f}, p_noise={p_noise:.5f}, rep {rep+1}/{repetitions} done")

# Save data
np.save("data/voter/phase/final_avg_opinions_phase_sweep.npy", final_avg_opinions)
print("Phase sweep complete and data saved.")
