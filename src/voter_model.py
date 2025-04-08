import numpy as np

def initialize_opinions(N_agents):
    """Initialize agents with random opinions (-1 and 1)."""
    return np.random.choice([-1,1], size=N_agents)

def random_connection_matrix(N_agents, p):
    """Generate a symmetric random adjacency matrix where each edge (i,j) exists with probability p."""
    
    # Initialize matrix (default: no connections)
    connection_matrix   = np.zeros(shape=(N_agents,N_agents), dtype=int)
    
    # Fill upper triangle (i < j) to avoid duplicate work
    for i in range(N_agents):
        # Skip self-connections (i != j)
        for j in range(i+1,N_agents):
            if np.random.rand() < p:
                connection_matrix[i,j] = 1
                connection_matrix[j,i] = 1
    
    return connection_matrix

# def voter_interaction(opinions, connection_matrix):
#     """One interaction: pick a random agent (i) and adopt a connected neighbor's (j) opinion."""

#     N_agents    = len(opinions)     # Number of agents
#     opinion_sum = np.sum(opinions)  # Sum of all opinions

#     # Check for consensus
#     if np.abs(opinion_sum) != N_agents:

#         # Find an agent 'i' with at least one disagreeing neighbor
#         while True:
#             i           = np.random.randint(0, N_agents)
#             neighbors   = np.where(connection_matrix[i]==1)[0]
#             N_neighbors = len(neighbors)
#             opinion_sum = np.sum(opinions[neighbors])

#             if (opinion_sum - opinions[i]*N_neighbors != 0):  
#                 break

#         # Choose a disagreeing neighbor 'j' and update opinion of 'i'
#         disagreeing_neighbors   = neighbors[opinions[neighbors] != opinions[i]]
#         j                       = np.random.choice(disagreeing_neighbors)
#         opinions[i]             = opinions[j]

#     return opinions

def voter_interaction(opinions, connection_matrix):
    """One interaction: pick a random agent (i) and adopt a connected neighbor's (j) opinion."""

    N_agents    = len(opinions)                         # Number of Agents
    i           = np.random.randint(0, N_agents)        # Sample agent i
    neighbors   = np.where(connection_matrix[i]==1)[0]  # Find all neighbors
    j           = np.random.choice(neighbors)           # Sample neighbor j
    opinions[i] = opinions[j]                           # Adopt Opinion

    return opinions

def average_opinion(opinions):
    """Average opinion (order parameter)."""

    return np.mean(opinions)

def interface_density(opinions, connection_matrix):
    """Fraction of connected edges where opinions differ."""

    disagreements   = 0
    n_agents        = len(opinions)

    # For each pair of agents check for opposing opinions
    for i in range(n_agents):
        for j in np.where(connection_matrix[i] == 1)[0]:
            if opinions[i] != opinions[j]:
                disagreements += 1

    total_connections = np.sum(connection_matrix)

    return disagreements / total_connections

if __name__ == '__main__':

    N_agents    = 100
    N_iter      = 10000
    p_con       = 0.05
    avg_opinion = np.zeros(N_iter)
    int_density = np.zeros(N_iter)
    opinions    = initialize_opinions(N_agents)
    con_matrix  = random_connection_matrix(N_agents,p_con)

    for i in range(N_iter):
        opinions        = voter_interaction(opinions, con_matrix)
        avg_opinion[i]  = average_opinion(opinions)
        int_density[i]  = interface_density(opinions, con_matrix)

    np.save(f"data/avg_opinion_p={p_con}", avg_opinion)
    np.save(f"data/int_density_p={p_con}", int_density)
    np.save(f"data/connection_matrix_p={p_con}", con_matrix)
