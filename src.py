# Credit: used Claude to create this code

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import seaborn as sns
from typing import List, Tuple, Dict
import pandas as pd


class ModelParameters:
    """Parameters for the evolutionary dynamics model"""
    def __init__(self, k: int, n_k: List[int], n_s: int, mu_r: float, 
                 mu_m: float, sigma: float, delta_t: float, c: np.ndarray,
                 alpha: np.ndarray, beta: np.ndarray, lambda_param: np.ndarray):
        self.k = k  # number of species
        self.n_k = n_k  # number of individuals per species
        self.n_s = n_s  # number of substances
        self.mu_r = mu_r  # Poisson rate parameter for mutations
        self.mu_m = mu_m  # mean of mutation effect
        self.sigma = sigma  # std of mutation effect
        self.delta_t = delta_t  # time step
        self.c = c  # cost matrix (n_s x k)
        self.alpha = alpha  # threshold parameters (n_s x k)
        self.beta = beta  # threshold parameters (n_s x k)
        self.lambda_param = lambda_param  # slope parameters (n_s x k)

class EvolutionaryDynamics:
    """
    Vectorized implementation of the evolutionary dynamics model.
    Uses numpy arrays for efficient computation.
    """
    
    def __init__(self, params: ModelParameters):
        self.params = params
        self.k = params.k
        self.n_k = np.array(params.n_k)
        self.n_s = params.n_s
        self.total_individuals = np.sum(self.n_k)
        
        # Create species assignment array for each individual
        self.species_idx = np.repeat(np.arange(self.k), self.n_k)
        
        # Initialize gene matrices
        # Shape: (n_substances, total_individuals)
        self.x = np.random.exponential(0.1, (self.n_s, self.total_individuals))
        self.y = np.random.exponential(0.1, (self.n_s, self.total_individuals))
        
        # Precompute individual offsets for each species
        self.species_offsets = np.concatenate([[0], np.cumsum(self.n_k)[:-1]])
    
    def response_function(self, y_vals: np.ndarray, j: int) -> np.ndarray:
        """
        Vectorized computation of response function g_{j,k}(y) for substance j
        y_vals: array of y values for all individuals
        Returns: array of g values for all individuals
        """
        # Get parameters for substance j and all species
        alpha_j = self.params.alpha[j, self.species_idx]  # (total_individuals,)
        beta_j = self.params.beta[j, self.species_idx]
        lambda_j = self.params.lambda_param[j, self.species_idx]
        
        # Initialize result array
        g_vals = np.ones_like(y_vals)
        
        # Apply piecewise function
        # Region 2: alpha <= y <= beta
        mask2 = (y_vals >= alpha_j) & (y_vals <= beta_j)
        g_vals[mask2] = 1.0 - lambda_j[mask2] * (y_vals[mask2] - alpha_j[mask2])
        
        # Region 3: y > beta
        mask3 = y_vals > beta_j
        g_vals[mask3] = -1.0
        
        return g_vals
    
    def calculate_fitness(self) -> np.ndarray:
        """
        Vectorized fitness calculation for all individuals
        Returns: fitness array of shape (total_individuals,)
        """
        # Calculate total substance production T_j for each substance
        T = np.sum(self.x, axis=1)  # Shape: (n_s,)
        
        # Initialize fitness array
        fitness = np.ones(self.total_individuals)
        
        # Calculate production costs for each individual
        # c has shape (n_s, k), we need to broadcast to (n_s, total_individuals)
        c_expanded = self.params.c[:, self.species_idx]  # (n_s, total_individuals)
        production_costs = np.sum(c_expanded * self.x, axis=0)  # (total_individuals,)
        
        # Calculate benefits from response to substances
        benefits = np.zeros(self.total_individuals)
        for j in range(self.n_s):
            g_vals = self.response_function(self.y[j, :], j)
            benefits += T[j] * g_vals
        
        fitness = 1.0 - production_costs + benefits
        return fitness
    
    def get_fitness_by_species(self, fitness: np.ndarray) -> Dict[int, np.ndarray]:
        """Convert flat fitness array to dictionary by species"""
        fitness_by_species = {}
        for k in range(self.k):
            start_idx = self.species_offsets[k]
            end_idx = start_idx + self.n_k[k]
            fitness_by_species[k] = fitness[start_idx:end_idx]
        return fitness_by_species
    
    def mutate_genes(self):
        """Vectorized mutation of all genes"""
        # Draw number of mutations for each gene of each individual
        P_shape = (self.n_s, self.total_individuals)
        P = poisson.rvs(self.params.mu_r * self.params.delta_t, size=P_shape)
        
        # Only mutate where P > 0
        mutation_mask = P > 0
        
        if np.any(mutation_mask):
            # Generate mutation effects where needed
            # For each position with mutations, generate P[i,j] random effects
            mutation_effects = np.zeros_like(self.x)
            
            for j in range(self.n_s):
                for i in range(self.total_individuals):
                    if P[j, i] > 0:
                        epsilon_sum = np.sum(np.random.normal(
                            self.params.mu_m, self.params.sigma, P[j, i]
                        ))
                        mutation_effects[j, i] = epsilon_sum
            
            # Apply mutations
            mutation_factors = np.exp(mutation_effects)
            self.x *= mutation_factors
            self.y *= mutation_factors
            
            # Clip to prevent extreme values
            self.x = np.clip(self.x, 1e-6, 1e6)
            self.y = np.clip(self.y, 1e-6, 1e6)
    
    def step(self):
        """Perform one time step"""
        self.mutate_genes()
    
    def run_simulation(self, n_steps: int, record_interval: int = 10) -> Dict:
        """Run simulation with vectorized operations"""
        history = {
            'time': [],
            'fitness_means': {k: [] for k in range(self.k)},
            'fitness_stds': {k: [] for k in range(self.k)},
            'x_means': {j: {k: [] for k in range(self.k)} for j in range(self.n_s)},
            'y_means': {j: {k: [] for k in range(self.k)} for j in range(self.n_s)},
            'total_substances': []
        }
        
        for step in range(n_steps):
            if step % record_interval == 0:
                # Calculate fitness
                fitness = self.calculate_fitness()
                fitness_by_species = self.get_fitness_by_species(fitness)
                
                history['time'].append(step * self.params.delta_t)
                
                # Record fitness statistics
                for k in range(self.k):
                    history['fitness_means'][k].append(np.mean(fitness_by_species[k]))
                    history['fitness_stds'][k].append(np.std(fitness_by_species[k]))
                
                # Record gene means by species
                for j in range(self.n_s):
                    for k in range(self.k):
                        start_idx = self.species_offsets[k]
                        end_idx = start_idx + self.n_k[k]
                        history['x_means'][j][k].append(np.mean(self.x[j, start_idx:end_idx]))
                        history['y_means'][j][k].append(np.mean(self.y[j, start_idx:end_idx]))
                
                # Record total substance production
                T = np.sum(self.x, axis=1)
                history['total_substances'].append(T.tolist())
            
            self.step()
        
        return history
    
    def get_final_analysis(self) -> Dict:
        """Get final state analysis"""
        fitness = self.calculate_fitness()
        fitness_by_species = self.get_fitness_by_species(fitness)
        
        # Calculate final production levels by species
        final_production = {}
        for j in range(self.n_s):
            for k in range(self.k):
                start_idx = self.species_offsets[k]
                end_idx = start_idx + self.n_k[k]
                final_production[(j,k)] = np.mean(self.x[j, start_idx:end_idx])
        
        analysis = {
            'final_fitness': fitness_by_species,
            'final_production': final_production,
            'fitness_summary': {k: {
                'mean': np.mean(fitness_by_species[k]),
                'std': np.std(fitness_by_species[k]),
                'min': np.min(fitness_by_species[k]),
                'max': np.max(fitness_by_species[k])
            } for k in range(self.k)}
        }
        
        return analysis

def analyze_production_regimes(params: ModelParameters, n_steps: int = 1000) -> Dict:
    """Vectorized analysis of production regimes"""
    model = EvolutionaryDynamics(params)
    history = model.run_simulation(n_steps)
    analysis = model.get_final_analysis()
    analysis['history'] = history
    return analysis

def plot_results(analysis: Dict, params: ModelParameters):
    """Plot simulation results from model"""
    history = analysis['history']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot fitness evolution
    axes[0,0].set_title('Fitness Evolution')
    for k in range(params.k):
        axes[0,0].plot(history['time'], history['fitness_means'][k], 
                      label=f'Species {k}', linewidth=2)
        axes[0,0].fill_between(history['time'], 
                              np.array(history['fitness_means'][k]) - np.array(history['fitness_stds'][k]),
                              np.array(history['fitness_means'][k]) + np.array(history['fitness_stds'][k]),
                              alpha=0.3)
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Fitness')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Plot substance production
    axes[0,1].set_title('Total Substance Production')
    substances = np.array(history['total_substances'])
    for j in range(params.n_s):
        axes[0,1].plot(history['time'], substances[:, j], 
                      label=f'Substance {j}', linewidth=2)
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Total Production')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Plot gene evolution (x genes)
    axes[1,0].set_title('Production Gene Evolution (x)')
    for j in range(min(2, params.n_s)):
        for k in range(params.k):
            axes[1,0].plot(history['time'], history['x_means'][j][k], 
                          label=f'Subst {j}, Sp {k}', linewidth=2)
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Mean x value')
    axes[1,0].legend()
    axes[1,0].grid(True)
    axes[1,0].set_yscale('log')
    
    # Plot fitness distributions
    axes[1,1].set_title('Final Fitness Distributions')
    for k in range(params.k):
        axes[1,1].hist(analysis['final_fitness'][k], alpha=0.6, 
                      label=f'Species {k}', bins=20)
    axes[1,1].set_xlabel('Fitness')
    axes[1,1].set_ylabel('Count')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()
    return fig
