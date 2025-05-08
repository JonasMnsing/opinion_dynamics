import numpy as np

def initialize_opinions(N_agents):
    """Initialize agents with random opinions (-1 and 1)
    as a 1D numpy array of length N_agents."""

    return None

def random_connection_matrix(N_agents, p):
    """Generate a symmetric random adjacency matrix of size (N_agents x N_agents)
    where each edge (i,j) exists with probability p."""
    
    return None

def voter_interaction(opinions, connection_matrix):
    """Sample an agent index (i) and one of its connected neighbors (j) at random.
    Let agent i adopt the opinion of agent j.
    """

    return opinions

# Simulation
if __name__ == '__main__':

    N_agents    = 50                                        # Number of Agents
    N_inter     = 1000                                      # Number of Interactions
    p_con       = 0.1                                       # Probability for a connection
    opinions    = initialize_opinions(N_agents)             # Initialize Opinions
    con_matrix  = random_connection_matrix(N_agents, p_con) # Initialize Network

    for i in range(N_agents):
        opinions    = voter_interaction(opinions, con_matrix)

    np.save("file_name.npy", opinions)