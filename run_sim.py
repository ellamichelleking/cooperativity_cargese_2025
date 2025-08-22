import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import seaborn as sns
import pandas as pd

from src import ModelParameters, analyze_production_regimes, plot_results

def create_parameters(k, n_individuals_per_species=50, n_s=3, rand_seed=0):
    """Helper function to create parameters for k species"""
    n_k = [n_individuals_per_species] * k
    
    # Generate random parameters for all species from a uniforn distribution
    np.random.seed(rand_seed)  # For reproducibility
    c = np.random.normal(0.1, 0.3, (n_s, k))
    alpha = np.random.uniform(0.2, 0.6, (n_s, k))
    beta = alpha + np.random.uniform(0.3, 0.8, (n_s, k))  # beta > alpha
    lambda_param = np.random.uniform(1.0, 3.0, (n_s, k))
    
    return ModelParameters(
        k=k, n_k=n_k, n_s=n_s,
        mu_r=0.1, mu_m=0.0, sigma=0.05, delta_t=0.1,
        c=c, alpha=alpha, beta=beta, lambda_param=lambda_param
    )


def run(k=10, n_k=50, rand_seed=0, n_steps=2000):
    params = create_parameters(k, n_individuals_per_species=n_k, n_s=3, rand_seed=rand_seed)
    
    print("Running evolutionary dynamics simulation...")
    print(f"Parameters: {params.k} species, {params.n_s} substances")
    
    analysis = analyze_production_regimes(params, n_steps=n_steps)
    
    print("\nFinal fitness summary:")
    for k in range(params.k):
        summary = analysis['fitness_summary'][k]
        print(f"Species {k}: mean={summary['mean']:.3f}, std={summary['std']:.3f}")
    
    print("\nFinal production levels:")
    for j in range(params.n_s):
        for k in range(params.k):
            prod = analysis['final_production'][(j,k)]
            print(f"Substance {j}, Species {k}: {prod:.3f}")
    
    plot_results(analysis, params)
    return analysis, params



if __name__ == "__main__":    
    # Run example simulation
    analysis, params = example_simulation(k=10, n_k=50)
