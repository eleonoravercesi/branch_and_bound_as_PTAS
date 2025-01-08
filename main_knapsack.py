'''
Experiments for the multidimensional knapsack
'''
import numpy as np
from exact_models.multi_knapsack import solve_multi_knapsack

np.random.seed(42)

n_knapsacks = 2
n_items = 10

profits = np.random.randint(1, 100, n_items)
weights = np.random.randint(1, 100, n_items)
capacities = np.random.randint(1, 100, n_knapsacks)

total_profit, assignment, status, runtime = solve_multi_knapsack(profits, weights, capacities)