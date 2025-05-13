import numpy as np
import networkx as nx
import multiprocessing as mp
from typing import Callable, Optional, Dict, Any, List, Tuple

class OpinionDynamicsModel:
    """Opinion dynamics model on networks.
    """
    def __init__(self, N_agents: int, graph_fn: Callable[...,np.ndarray],
                 graph_kwargs: Optional[Dict[str, Any]] = None, p_noise: float = 0.0):
        """
        Parameters
        ----------
        N_agents
            Number of agents (nodes).
        graph_fn
            Function to generate an adjacency matrix, e.g., connected_erdos_renyi, small_world_graph.
        graph_kwargs
            Keyword arguments passed to graph_fn.
        p_noise
            Probability of spontaneous opinion flip at each update.
        """
        self.N                  = N_agents
        self.graph_fn           = graph_fn
        self.graph_kwargs       = graph_kwargs or {}
        self.p_noise            = p_noise
        self.connection_matrix  = self.graph_fn(self.N, **self.graph_kwargs)
        self.opinions           = self.initialize_opinions()
    
    def initialize_opinions(self) -> np.ndarray:
        """Initialize agents with random opinions (-1 or +1)."""
        return np.random.choice([-1, 1], size=self.N)
    
    def voter_interaction(self) -> None:
        """Perform one asynchronous voter update with noise."""
        i = np.random.randint(0, self.N)
        if np.random.rand() < self.p_noise:
            self.opinions[i] *= -1
        else:
            neighbors = np.where(self.connection_matrix[i] == 1)[0]
            if neighbors.size > 0:
                j = np.random.choice(neighbors)
                self.opinions[i] = self.opinions[j]
    
    def average_opinion(self) -> float:
        """Compute the global mean opinion (magnetization)."""
        return float(np.mean(self.opinions))
    
    def interface_density(self) -> float:
        """Fraction of edges connecting opposite opinions."""
        # Count each edge twice in adjacency matrix, so divide by total connections
        conflicts = 0
        total = np.sum(self.connection_matrix)
        for i in range(self.N):
            for j in np.where(self.connection_matrix[i] == 1)[0]:
                if self.opinions[i] != self.opinions[j]:
                    conflicts += 1
        return conflicts / total if total > 0 else 0.0
    
    def run_trajectory(self, max_steps: int = 100000, stop_on_consensus: bool = True) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Run a single dynamics trajectory.

        Returns
        -------
        m_list
            Array of average opinion over time.
        rho_list
            Array of interface density over time.
        t
            Time-step at which simulation stopped.
        """
        # Reset opinions
        self.opinions           = self.initialize_opinions()
        m_list: List[float]     = []
        rho_list: List[float]   = []

        for t in range(1, max_steps + 1):
            self.voter_interaction()
            m_list.append(self.average_opinion())
            rho_list.append(self.interface_density())
            if stop_on_consensus and abs(m_list[-1]) == 1.0:
                break

        return np.array(m_list), np.array(rho_list), t
    
    def _worker_run(self, args: Tuple[int, bool]) -> Tuple[np.ndarray,np.ndarray,int]:
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
        times: List[int]            = []
        m_trajs: List[np.ndarray]   = []
        rho_trajs: List[np.ndarray] = []
        
        for m, rho, t in results:
            m_trajs.append(m)
            rho_trajs.append(rho)
            times.append(t)

        return {
            'times': times,
            'mean_time': float(np.mean(times)),
            'm_trajs': m_trajs,
            'rho_trajs': rho_trajs
        }
    
def connected_erdos_renyi(N_agents: int, p: float) -> np.ndarray:
    """Generate a connected Erdős–Rényi graph adjacency matrix."""
    while True:
        G = nx.erdos_renyi_graph(N_agents, p)
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