# Modell- und datenbasierte Methoden in den Naturwissenschaften
### **General Remarks**
- **Learning Objective**: Project oriented working with Python
- **Weekly Meetings**:
  - Discussions on theory and modelling
  - Presentation and discussion of your results
  - Discussion of your questions
- **Midterm Presentation**: 03.06.2025, 14:15
  - Overview / Theory / Motivation
  - Baseline model
  - Outlook for the second half of the semester
- **Final Presentation**: 10.07.2025, 16:15
  - Recap of last presentation / theory
  - Model adjustments
  - Results

### **Opinions Dynamics / Voter Model**
- Review Paper on Social Dynamics: https://arxiv.org/abs/0710.3256
- Physical Correspondance of the Voter Model: Ising Model
- Collection of $N$ spins (agents) $s_i$ that can assume two values $\{-1,1\}$
- Each spin is pushed to be aligned with its nearest neighbors disturbed by noise
- The total energy of the system is defied via the sum over all nearest neighbors for each spin:
$$H = \frac{1}{2}\sum_{<i,j>}s_is_j$$
- In a given iteration a spin flips given the Boltzman distribution
$$p \propto \exp(-\Delta H / k_BT)$$
- For small temperatures, long range order is established $\rightarrow$ Ferromagnet
- For temperatures above a critical temperature $T_C$ (Curie Temperature), the system remains macroscopically disordered. 
- Transition point is characterized by the magnetiztaion:
$$m = \frac{1}{N}\sum_i <s_i>$$
- For our Voter model, the spin of agent $i$ now corresponds to the opinion of this particular agent
- We can further simplify the simulation for now, assuming that we select an agent $i$ at random and one of its nearest neighbors $j$. Agent $i$ will then just adopt the opinion of $j$.