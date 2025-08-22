import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import seaborn as sns
import pandas as pd
import argparse

from src import ModelParameters, EvolutionaryModel, plot_simulation_results

parser = argparse.ArgumentParser()
parser.add_argument('-k','--k_species', default=10)
parser.add_argument('-ni','--n_individuals', description='Number of individuals per species', default=50)
parser.add_argument('-nj','--n_substances', default=4)
parser.add_argument('-mr','--mutation_rate', description='Poisson rate for number of mutations', default=0.05)
parser.add_argument('-mu','--mean_mutation', description='mean of mutation effect', default=0.0)
parser.add_argument('-std','--sigma_mutation', description='standard deviation of mutation effect', default=0.02)
parser.add_argument('-dt','--timestep', default=1.0)
parser.add_argument('-ns','--num_steps', default=1000)
parser.add_argument('-se','--save_every', default=10)

args = vars(parser.parse_args())

k = args['k_species']
n_i = args['n_individuals']
n_j = args['n_substances']
mu_r = args['mutation_rate']
mu_m = args['mean_mutation']
sigma = args['sigma_mutation']
dt = args['timestep']
num_steps = args['num_steps']
save_every = args['save_every']


if __name__ == "__main__":
    # Test with large number of species
    params = ModelParameters(
        k_species=k,  # Large number of species
        n_individuals=[n_i] * k,  # n_i individuals per species
        n_substances=n_j,
        mu_r=mu_r,
        mu_m=mu_m,
        sigma=sigma,
        dt=dt
    )
    
    print(f"Running simulation with {params.k_species} species...")
    model = EvolutionaryModel(params)
    history = model.run_simulation(n_steps=num_steps, save_interval=save_every)
    
    print("Plotting results...")
    plot_simulation_results(history, params)
    
    # Analyze parameter regimes
    print("Analyzing mutation rate regimes...")
    base_params = ModelParameters(k_species=k, n_substances=n_j)
    mutation_analysis = analyze_parameter_regime(
        base_params, 'mu_r', [0.01, 0.05, 0.1, 0.2, 0.5], 
        n_steps=num_steps, n_replicates=3
    )
    
    print("\nMutation rate analysis summary:")
    summary = mutation_analysis.groupby('parameter_value').agg({
        'final_production': ['mean', 'std'],
        'final_fitness': ['mean', 'std']
    }).round(4)
    print(summary)
