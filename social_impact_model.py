import numpy as np
import networkx as nx

def connected_erdos_renyi(N_agents: int, p: float) -> np.ndarray:
    """Generate a connected Erdős–Rényi graph adjacency matrix, with iteration limit."""
    attempt = 0
    while True:
        if attempt >= 100000:
            raise RuntimeError(f"Failed to generate connected ER graph after {attempt} attempts")
        G = nx.erdos_renyi_graph(N_agents, p)
        attempt += 1
        if nx.is_connected(G):
            break
    return nx.to_numpy_array(G, dtype=int)

def compute_distances_and_weights(connection_matrix, N_agents, alpha=2, cutoff=10):
    """Return (dist_matrix, weight_matrix) for a given adjacency."""
    
    # Build a graph from the connection matrix
    G       = nx.from_numpy_array(connection_matrix)

    # Init distances between agent i and j as Inf
    dist    = np.full((N_agents, N_agents), np.inf)

    # For each shortest paths between i and j given a cutoff
    for i, lengths in nx.all_pairs_shortest_path_length(G, cutoff=cutoff):
        for j, d in lengths.items():
            # Put shortest path distance into distance array
            dist[i, j] = d

    # The distance between agent i and i is Inf
    np.fill_diagonal(dist, np.inf)

    # Calculate weights as 1 / d^alpha
    w = 1.0 / (dist**alpha)

    # Weights which are Inf are set Zero
    w[np.isinf(w)] = 0.0

    return w

def social_impact_interaction(opinions, weights, pressure, support, beta=1.0, p_noise=0.1):
    """Sample an agent index (i) and one of its connected neighbors (j) at random.
    Update opinion according to the social impact rule.
    """

    N   = opinions.size         # Number of agents
    i   = np.random.randint(N)  # Sample an agent
    x_i = opinions[i]           # Get its opinion  


    w_row = weights[i,:]        # All weights of i and his neighborhood

    # Social impact
    p_term  = np.sum(w_row * pressure * (1 - x_i * opinions))
    s_term  = np.sum(w_row * support  * (1 + x_i * opinions))
    F_i     = x_i*(p_term - s_term)

    # New Opinion of i based on impact and noise
    opinions[i] = -np.tanh(beta * F_i + np.random.uniform(-p_noise, p_noise))

    return opinions

# Simulation
if __name__ == '__main__':

    N_agents    = 80    # Number of Agents
    N_runs      = 50    # Number of runs
    N_inter     = 1000  # Number of Interactions
    p_con       = 0.1   # Probability for a connection
    beta        = 1.0   # beta inside tanh function
    p_noise     = 0.1   # Noise boundary
    o_list      = []    # Container for each run

    for n in range(N_runs):
        # Inside each run we need to reinitialize our states
        opinions    = np.random.uniform(-1,1,N_agents)                      # Initialize Opinions
        support     = np.random.uniform(-1,1,N_agents)                      # Initialize Support
        pressure    = np.random.uniform(-1,1,N_agents)                      # Initialize Pressure
        con_matrix  = connected_erdos_renyi(N_agents, p_con)                # Initialize Network
        weights     = compute_distances_and_weights(con_matrix, N_agents)   # Pre compute weights for speed
        o_runs      = []                                                    # Container for opinions in time given a run

        for i in range(N_inter):
            opinions = social_impact_interaction(opinions, weights, pressure, support, beta, p_noise)
            o_runs.append(opinions.copy()) # important to use copy!!! 
        
        o_list.append(o_runs)

    np.save("opinions.npy", o_list)