import numpy as np
import networkx as nx

def initialize_opinions(N_agents):
    """Initialize agents with random opinions (-1 and 1)."""
    return np.random.choice([-1,1], size=N_agents)

def connected_erdos_renyi(N_agents, p):
    """Generate a random adjacency matrix from a binomial graph with probability p for an edge connection"""
    while True:
        G = nx.erdos_renyi_graph(N_agents, p)
        if nx.is_connected(G):
            break
    return nx.to_numpy_array(G, dtype=int)

def small_world_graph(N_agents, k=4, beta=0.1):
    G = nx.watts_strogatz_graph(N_agents, k, beta)
    return nx.to_numpy_array(G, dtype=int)

def scale_free_graph(N, m=2):
    G = nx.barabasi_albert_graph(N, m)
    return nx.to_numpy_array(G, dtype=int)

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

def voter_interaction(opinions, connection_matrix, p_noise=0.0):
    """One interaction: pick a random agent (i) and adopt a connected neighbor's (j) opinion."""

    N_agents    = len(opinions)                         # Number of Agents
    i           = np.random.randint(0, N_agents)        # Sample agent i
    
    if np.random.rand() < p_noise:
        opinions[i] *= -1
    else:
        neighbors   = np.where(connection_matrix[i]==1)[0]  # Find all neighbors
        if len(neighbors) > 0:
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
    p_con       = 0.1
    p_noise     = 0.1
    avg_opinion = np.zeros(N_iter)
    int_density = np.zeros(N_iter)
    opinions    = initialize_opinions(N_agents)
    con_matrix  = connected_erdos_renyi(N_agents,p_con)

    for i in range(N_iter):
        opinions        = voter_interaction(opinions, con_matrix, p_noise)
        avg_opinion[i]  = average_opinion(opinions)
        int_density[i]  = interface_density(opinions, con_matrix)

    np.save(f"data/avg_opinion", avg_opinion)
    np.save(f"data/int_density", int_density)
    np.save(f"data/connection_matrix", con_matrix)
