import numpy as np
import networkx as nx
import multiprocessing as mp
from typing import Callable, Optional, Dict, Any, Tuple

class OpinionDynamicsModel:
    """A continuous-opinion social impact model on networks, including distance-weighted influence.
    Discrete (voter-like) behavior is recovered with beta=np.inf and pressure/support=ones.
    Influence decays as 1/d_{ij}^alpha over the shortest-path distance.
    """
    def __init__(
            self,
            N_agents: int,
            graph_fn: Callable[...,np.ndarray],
            graph_kwargs: Optional[Dict[str, Any]] = None,
            p_noise: float = 0.0,
            pressure: Optional[np.ndarray] = None,
            support: Optional[np.ndarray] = None,
            beta: float = 1.0,
            alpha: float = 2.0,
            distance_cutoff: int = 10):
        """
        Parameters
        ----------
        N_agents
            Number of agents (nodes).
        graph_fn
            Function to generate an adjacency matrix, e.g., connected_erdos_renyi, small_world_graph.
        graph_kwargs
            Keyword arguments for graph_fn.
        p_noise
            Noise range: uniform in [-p_noise,p_noise].
        pressure, support
            Arrays of persuasiveness/supportiveness. If None, defaults to ones.
        beta
            Gain parameter for tanh update. beta=np.inf recovers discrete flips.
        alpha
            Exponent for distance decay
        distance_cutoff
            Maximum graph-distance to include in the "social impact" sum.
            Any j with d_{ij} > distance_cutoff will be treated as if distance=inf (i.e. weight=0)
        """
        self.N              = N_agents
        self.graph_fn       = graph_fn
        self.graph_kwargs   = graph_kwargs or {}
        self.p_noise        = p_noise
        self.alpha          = alpha
        self.beta           = beta

        # Build adjacency matrix
        self.connection_matrix  = self.graph_fn(self.N, **self.graph_kwargs)

        # Build a NetworkX graph from adjacency
        G = nx.from_numpy_array(self.connection_matrix)
        
        # Precompute all-pairs shortest path legnths (with cutoff) as a 2D array
        self.dist_matrix = np.full((self.N, self.N), np.inf)
        for i, lengths in nx.all_pairs_shortest_path_length(G,cutoff=distance_cutoff):
            for j, d in lengths.items():
                self.dist_matrix[i,j] = d
        # Ensure self-distance is infinite (so we skip i->i)
        np.fill_diagonal(self.dist_matrix, np.inf)

        # Set pressure and support arrays
        self.pressure   = pressure if pressure is not None else np.ones(N_agents)
        self.support    = support if support is not None else np.ones(N_agents)
        
        # Initialize continuous opinions in [-1,1]            
        self.opinions   = self.initialize_opinions()
    
    def initialize_opinions(self) -> np.ndarray:
        """Draw random continuous opinions in [-1 or +1]."""
        return np.random.uniform(-1, 1, size=self.N)
    
    def agent_interaction(self) -> None:
        """Perform one asynchronous update using tanh beta-gain."""

        # Sample an agent i at random
        i       = np.random.randint(0, self.N)
        x_i     = self.opinions[i]
        x_vec   = self.opinions

        # Distance based weights
        dists       = self.dist_matrix[i,:]
        weights     = 1/(dists**self.alpha)

        # Social impcat on agent i
        pressure_part   = np.sum(weights * self.pressure * (1 - x_i * x_vec))
        support_part    = np.sum(weights * self.pressure * (1 + x_i * x_vec))
        social_impact   = pressure_part - support_part

        # Add a random field
        h   = np.random.uniform(-self.p_noise,self.p_noise)
        F_i = x_i * social_impact + h

        if np.isinf(self.beta):
            self.opinions[i] = np.sign(F_i)
        else:
            self.opinions[i] = np.tanh(self.beta * F_i)
    
    def average_opinion(self) -> float:
        """Compute the global mean opinion (magnetization)."""
        return float(np.mean(self.opinions))
    
    def spatial_correlation(self) -> np.ndarray:
        """
        Compute spatial correlation C(d) = ⟨ x_i * x_j ⟩ averaged over all pairs (i, j)
        whose shortest-path distance dist_matrix[i,j] == d (finite).

        Returns
        -------
        corrs
            Array of spatial correlations
        """

        # Determine maximum finite distance
        finite_dists    = self.dist_matrix[np.isfinite(self.dist_matrix)]
        max_d           = int(finite_dists.max())

        # Precompute outer product of opinions
        products = self.opinions[:,None]*self.opinions[None,:]
        
        corrs = []
        corrs.append(1.0)
        for d in range(1, max_d+1):
            mask    = (self.dist_matrix==d)
            values  = products[mask]
            if values.size == 0:
                corrs.append(0.0)
            else:
                corrs.append(float(np.mean(values)))
        return np.array(corrs)
        
    def run_trajectory(self, max_steps: int = 100000, stop_on_consensus: bool = True) -> Tuple[np.ndarray, int]:
        """
        Run a single dynamics trajectory.

        Returns
        -------
        m_list
            Array of average opinion over time.
        t
            Time-step at which simulation stopped.
        """
        # Reset opinions
        self.opinions   = self.initialize_opinions()
        m_list          = []

        for t in range(1, max_steps + 1):
            self.agent_interaction()
            m_list.append(self.average_opinion())
            if stop_on_consensus and abs(m_list[-1]) == 1.0:
                break

        return np.array(m_list), t
    
    def _worker_run(self, args: Tuple[int, bool]) -> Tuple[np.ndarray,int]:
        """Helper for multiprocessing: runs one trajectory."""
        max_steps, stop_on_consensus = args
        return self.run_trajectory(max_steps, stop_on_consensus)
    
    def ensemble_stats(self, n_runs: int = 50, max_steps: int = 100000,
                       stop_on_consensus: bool = True, n_processes: Optional[int] = None) -> Dict[str, Any]:
        """
        Run multiple independent trajectories (optionally in parallel) and collect statistics.

        Parameters
        ----------
        n_runs
            Number of independent runs
        max_steps
            Maximum steps per run
        stop_on_consensus
            Whether to stop each run on full consensus
        n_processes
            Number of processes for parallel execution; if None, runs serially.
        """
        # Prepare arguments for each run
        args_list = [(max_steps, stop_on_consensus) for _ in range(n_runs)]

        if n_processes and n_processes > 1:
            with mp.Pool(processes=n_processes) as pool:
                results = pool.map(self._worker_run, args_list)
        else:
            results = list(map(self._worker_run, args_list))

        # Unpack results
        times       = []
        m_trajs     = []

        for m, t in results:
            m_trajs.append(m)
            times.append(t)

        return {
            'times': times,
            'm_trajs': m_trajs,
        }
    
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


def small_world_graph(N_agents: int, k: int = 4, beta: float = 0.1) -> np.ndarray:
    """Generate a Watts–Strogatz small-world graph adjacency matrix."""
    G = nx.watts_strogatz_graph(N_agents, k, beta)
    return nx.to_numpy_array(G, dtype=int)


def scale_free_graph(N_agents: int, m: int = 2) -> np.ndarray:
    """Generate a Barabási–Albert scale-free graph adjacency matrix."""
    G = nx.barabasi_albert_graph(N_agents, m)
    return nx.to_numpy_array(G, dtype=int)