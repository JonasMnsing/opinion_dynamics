import numpy as np

def initialize_opinions(N_agents):
    """Initialize agents with random opinions (-1 and 1)."""
    return np.random.choice([-1,1], size=N_agents)

def initialize_properties(N_agents):
    """Initialize agent properties persuasiveness and suportiveness
    """
    p   = np.random.rand(N_agents)
    s   = np.random.rand(N_agents)

    return p, s

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

# def nearest_neighbor_impact(impact, opinions, pressure, support, connection_matrix):
#     """Calculate the social impact on each agent
#     """
    
#     for i,val_i in enumerate(opinions):
#         pressure_part   = 0
#         support_part    = 0
#         neighbors       = np.where(connection_matrix[i]==1)[0]

#         for j in neighbors:
#             pressure_part   += pressure[j]*(1-val_i*opinions[j])
#             support_part    += support[j]*(1+val_i*opinions[j])

#         impact[i]   = pressure_part - support_part

#     return impact

def nearest_neighbor_impact_interaction(opinions, pressure, support, connection_matrix, noise=0.01):
    """Calculates social impact for one agent and updates its opiniong given impact and noise
    """

    N_agents        = len(opinions)
    i               = np.random.randint(0, N_agents)
    neighbors       = np.where(connection_matrix[i]==1)[0]
    pressure_part   = 0
    support_part    = 0

    for j in neighbors:
        pressure_part   += pressure[j]*(1-opinions[i]*opinions[j])
        support_part    += support[j]*(1+opinions[i]*opinions[j])

    h           = np.random.uniform(-noise, noise)
    impact      = pressure_part - support_part
    arg         = opinions[i]*impact + h
    opinions[i] = -np.sign(arg)

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
    N_iter      = 200
    p_con       = 0.1
    noise       = 0.0
    avg_opinion = np.zeros(N_iter)
    int_density = np.zeros(N_iter)
    opinion_arr = np.zeros((N_iter,N_agents))
    opinions    = initialize_opinions(N_agents)
    p, s        = initialize_properties(N_agents)
    con_matrix  = random_connection_matrix(N_agents,p_con)

    for i in range(N_iter):
        opinions            = nearest_neighbor_impact_interaction(opinions, p, s, con_matrix, noise)
        avg_opinion[i]      = average_opinion(opinions)
        int_density[i]      = interface_density(opinions, con_matrix)
        opinion_arr[i,:]    = opinions

    np.save(f"data/social_impact_avg_opinion", avg_opinion)
    np.save(f"data/social_impact_int_density", int_density)
    np.save(f"data/social_impact_connection_matrix", con_matrix)
    np.save(f"data/social_impact_opinions", opinion_arr)
