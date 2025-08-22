# Credit: used Claude to create this code

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd

@dataclass
class ModelParameters:
    """Parameters for the evolutionary model"""
    k_species: int = 5  # number of species
    n_individuals: List[int] = None  # individuals per species
    n_substances: int = 3  # number of substances
    
    # Mutation parameters
    mu_r: float = 0.1  # Poisson rate for number of mutations
    mu_m: float = 0.0  # mean of mutation effect
    sigma: float = 0.01  # std of mutation effect
    dt: float = 1.0  # time step
    
    # Fitness-secretion coupling
    secretion_fitness_coupling: bool = True  # Whether secretion is proportional to fitness
    
    # Cost parameters (k_species x n_substances)
    costs: np.ndarray = None  # c_{j,k}
    
    # Response function parameters (k_species x n_substances for each)
    alpha: np.ndarray = None  # threshold parameters
    beta: np.ndarray = None   # upper threshold parameters  
    lambda_param: np.ndarray = None  # slope parameters
    
    def __post_init__(self):
        if self.n_individuals is None:
            self.n_individuals = [100] * self.k_species
            
        if self.costs is None:
            # Default: random costs between 0.01 and 0.1
            self.costs = np.random.uniform(0.01, 0.1, (self.k_species, self.n_substances))
            
        if self.alpha is None:
            # Default: random thresholds
            self.alpha = np.random.uniform(0.1, 0.5, (self.k_species, self.n_substances))
            
        if self.beta is None:
            # Default: beta > alpha
            self.beta = self.alpha + np.random.uniform(0.2, 0.8, (self.k_species, self.n_substances))
            
        if self.lambda_param is None:
            # Default: random slopes
            self.lambda_param = np.random.uniform(0.5, 2.0, (self.k_species, self.n_substances))

class EvolutionaryModel:
    """Vectorized implementation of the multi-species evolutionary model"""
    
    def __init__(self, params: ModelParameters):
        self.params = params
        self.reset_population()
        
    def reset_population(self):
        """Initialize population with random genes"""
        self.x_genes = []  # excretion genes
        self.y_genes = []  # control genes
        
        for k in range(self.params.k_species):
            n_k = self.params.n_individuals[k]
            # Initialize genes with small positive values
            # x_k = np.random.lognormal(mean=-1, sigma=0.5, 
            #                         size=(n_k, self.params.n_substances))
            x_k = np.zeros((n_k, self.params.n_substances))
            y_k = np.random.lognormal(mean=-1, sigma=0.5,
                                    size=(n_k, self.params.n_substances))
            self.x_genes.append(x_k)
            self.y_genes.append(y_k)
    
    def mutate_genes(self):
        """Apply mutations to all genes using vectorized operations"""
        for k in range(self.params.k_species):
            n_k = self.params.n_individuals[k]
            
            # Number of mutations per individual per gene (Poisson)
            n_mutations = np.random.poisson(
                self.params.mu_r * self.params.dt, 
                size=(n_k, self.params.n_substances)
            )
            
            # Generate mutation effects
            for i in range(n_k):
                for j in range(self.params.n_substances):
                    if n_mutations[i, j] > 0:
                        # Sum of mutation effects
                        mutation_sum = np.sum(np.random.normal(
                            self.params.mu_m, self.params.sigma, n_mutations[i, j]
                        ))
                        
                        # Apply multiplicative mutations
                        self.x_genes[k][i, j] += mutation_sum
                        self.y_genes[k][i, j] += mutation_sum
    
    def compute_response_function(self, y_values: np.ndarray, k: int) -> np.ndarray:
        """Compute g_{j,k} response function vectorized over substances"""
        g = np.ones_like(y_values)
        
        for j in range(self.params.n_substances):
            alpha_jk = self.params.alpha[k, j]
            beta_jk = self.params.beta[k, j]
            lambda_jk = self.params.lambda_param[k, j]
            
            # Linear regime: alpha <= y <= beta
            linear_mask = (y_values[:, j] >= alpha_jk) & (y_values[:, j] <= beta_jk)
            g[linear_mask, j] = 1 - lambda_jk * (y_values[linear_mask, j] - alpha_jk)
            
            # Negative regime: y > beta
            negative_mask = y_values[:, j] > beta_jk
            g[negative_mask, j] = -1
            
        return g
    
    def compute_fitness_and_substances(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Compute fitness and total substances with fitness-dependent secretion.
        This requires iteration since fitness affects secretion which affects fitness.
        """
        # Initial guess for fitness (without response term)
        fitness_by_species = []
        for k in range(self.params.k_species):
            # Initial fitness is just 1 - cost term
            cost_term = -np.sum(self.params.costs[k, :] * self.x_genes[k], axis=1)
            initial_fitness = 1 + cost_term
            fitness_by_species.append(initial_fitness)
        
        # Iterate to find consistent fitness and secretion levels
        for iteration in range(5):  # Usually converges quickly
            # Compute actual secreted amounts (fitness-dependent)
            T = np.zeros(self.params.n_substances)
            for k in range(self.params.k_species):
                fitness = fitness_by_species[k]
                # Actual secretion = x_gene * fitness (fitness-proportional secretion)
                actual_secretion = np.exp(self.x_genes[k]) * fitness[:, np.newaxis]
                T += np.sum(actual_secretion, axis=0)
            
            # Update fitness with response term
            new_fitness_by_species = []
            for k in range(self.params.k_species):
                
                # Cost term (based on actual secrtion)
                actual_secretion = np.exp(self.x_genes[k]) * fitness[:, np.newaxis]
                cost_term = -np.sum(self.params.costs[k, :] * actual_secretion, axis=1)
                
                # Response term (based on total actual secretions T)
                g_values = self.compute_response_function(self.y_genes[k], k)
                response_term = np.sum(T * g_values, axis=1)
                
                # Total fitness
                fitness = 1 + cost_term + response_term
                new_fitness_by_species.append(fitness)
            
            fitness_by_species = new_fitness_by_species
        
        return fitness_by_species, T
    
    def compute_fitness(self) -> List[np.ndarray]:
        """Compute fitness (wrapper for backward compatibility)"""
        fitness_by_species, _ = self.compute_fitness_and_substances()
        return fitness_by_species
    
    def compute_total_substances(self) -> np.ndarray:
        """Compute T_j with fitness-dependent secretion"""
        _, T = self.compute_fitness_and_substances()
        return T
    
    def wright_fisher_selection(self, fitness_by_species: List[np.ndarray]):
        """Implement Wright-Fisher selection"""
        new_x_genes = []
        new_y_genes = []
        
        for k in range(self.params.k_species):
            n_k = self.params.n_individuals[k]
            fitness = fitness_by_species[k]
            
            # Avoid negative fitness (set minimum to small positive value)
            fitness = np.maximum(fitness, 1e-6)
            
            # Selection probabilities proportional to fitness
            probs = fitness / np.sum(fitness)
            
            # Sample new generation
            selected_indices = np.random.choice(n_k, size=n_k, p=probs)
            
            new_x_genes.append(self.x_genes[k][selected_indices].copy())
            new_y_genes.append(self.y_genes[k][selected_indices].copy())
        
        self.x_genes = new_x_genes
        self.y_genes = new_y_genes
    
    def step(self) -> Dict:
        """Single time step of the model"""
        # Compute fitness and substances before mutations
        fitness_before, T_before = self.compute_fitness_and_substances()
        
        # Apply mutations
        self.mutate_genes()
        
        # Compute fitness and substances after mutations
        fitness_after, T_after = self.compute_fitness_and_substances()
        
        # Selection
        self.wright_fisher_selection(fitness_after)
        
        return {
            'fitness_before': fitness_before,
            'fitness_after': fitness_after,
            'total_substances_before': T_before,
            'total_substances_after': T_after,
            'mean_fitness': [np.mean(f) for f in fitness_after],
            'std_fitness': [np.std(f) for f in fitness_after]
        }
    
    def run_simulation(self, n_steps: int, save_interval: int = 10) -> Dict:
        """Run simulation for multiple steps and collect data"""
        history = {
            'mean_fitness': [],
            'std_fitness': [],
            'total_substances': [],
            'mean_x_genes': [],
            'mean_y_genes': [],
            'production_levels': []
        }
        
        for step in range(n_steps):
            result = self.step()
            
            if step % save_interval == 0:
                history['mean_fitness'].append(result['mean_fitness'])
                history['std_fitness'].append(result['std_fitness'])
                history['total_substances'].append(result['total_substances_after'])  # Use post-mutation values
                
                # Mean gene values by species
                mean_x = [np.mean(x, axis=0) for x in self.x_genes]
                mean_y = [np.mean(y, axis=0) for y in self.y_genes]
                history['mean_x_genes'].append(mean_x)
                history['mean_y_genes'].append(mean_y)
                
                # Production levels (mean excretion per species)
                production = [np.mean(np.sum(x, axis=1)) for x in self.x_genes]
                history['production_levels'].append(production)
        
        return history

def analyze_parameter_regime(base_params: ModelParameters, 
                           parameter_name: str,
                           parameter_values: List[float],
                           n_steps: int = 1000,
                           n_replicates: int = 5) -> pd.DataFrame:
    """Analyze model behavior across parameter regimes"""
    results = []
    
    for param_val in parameter_values:
        for replicate in range(n_replicates):
            # Create modified parameters
            params = ModelParameters(**base_params.__dict__)
            setattr(params, parameter_name, param_val)
            
            # Run simulation
            model = EvolutionaryModel(params)
            history = model.run_simulation(n_steps)
            
            # Extract final values
            final_production = np.mean(history['production_levels'][-10:], axis=0)
            final_fitness = np.mean(history['mean_fitness'][-10:], axis=0)
            final_substances = np.mean(history['total_substances'][-10:], axis=0)
            
            for species in range(params.k_species):
                results.append({
                    'parameter_value': param_val,
                    'replicate': replicate,
                    'species': species,
                    'final_production': final_production[species],
                    'final_fitness': final_fitness[species],
                    'total_substance_0': final_substances[0] if len(final_substances) > 0 else 0,
                    'total_substance_1': final_substances[1] if len(final_substances) > 1 else 0,
                    'total_substance_2': final_substances[2] if len(final_substances) > 2 else 0,
                })
    
    return pd.DataFrame(results)

def plot_simulation_results(history: Dict, params: ModelParameters):
    """Plot simulation results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    steps = np.arange(len(history['mean_fitness']))
    
    # Mean fitness over time
    mean_fitness = np.array(history['mean_fitness'])
    axes[0, 0].plot(steps, mean_fitness)
    axes[0, 0].set_title('Mean Fitness by Species')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Mean Fitness')
    axes[0, 0].legend([f'Species {k}' for k in range(params.k_species)])
    
    # Total substances over time
    total_substances = np.array(history['total_substances'])
    axes[0, 1].plot(steps, total_substances)
    axes[0, 1].set_title('Total Substances')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Total Amount')
    axes[0, 1].legend([f'Substance {j}' for j in range(params.n_substances)])
    
    # Production levels by species
    production = np.array(history['production_levels'])
    axes[0, 2].plot(steps, production)
    axes[0, 2].set_title('Production Levels by Species')
    axes[0, 2].set_xlabel('Time Step')
    axes[0, 2].set_ylabel('Mean Production')
    axes[0, 2].legend([f'Species {k}' for k in range(params.k_species)])
    
    # Final fitness distribution
    final_fitness = history['mean_fitness'][-1]
    axes[1, 0].bar(range(params.k_species), final_fitness)
    axes[1, 0].set_title('Final Mean Fitness')
    axes[1, 0].set_xlabel('Species')
    axes[1, 0].set_ylabel('Fitness')
    
    # Final substance totals
    final_substances = history['total_substances'][-1]
    axes[1, 1].bar(range(params.n_substances), final_substances)
    axes[1, 1].set_title('Final Total Substances')
    axes[1, 1].set_xlabel('Substance')
    axes[1, 1].set_ylabel('Total Amount')
    
    # Cost vs Production scatter
    final_production = history['production_levels'][-1]
    mean_costs = np.mean(params.costs, axis=1)
    axes[1, 2].scatter(mean_costs, final_production)
    axes[1, 2].set_title('Cost vs Production')
    axes[1, 2].set_xlabel('Mean Cost')
    axes[1, 2].set_ylabel('Final Production')
    for k in range(params.k_species):
        axes[1, 2].annotate(f'S{k}', (mean_costs[k], final_production[k]))
    
    plt.tight_layout()
    plt.show()
