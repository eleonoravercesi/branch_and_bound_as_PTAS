'''
Experiments for the multidimensional knapsack
'''
import numpy as np
from BeB.multi_knapsack import BranchAndBound
from exact_models.multi_knapsack import solve_multi_knapsack

np.random.seed(43)

n_knapsacks = 2
n_items = 5

profits = [2, 4,9,12,10]
weights = [1, 2,3,4,5]
capacities = [5,6]

total_profit, assignment, status, runtime = solve_multi_knapsack(profits, weights, capacities)
print(f"Total profit (exact): {total_profit}")


beb = BranchAndBound("greatest_upper_bound", "dantzig_upper_bound", "critical_element", "martello_toth_rule", 0.99)

LB, X_int, UB, runtime = beb.solve(profits, weights, capacities, verbose = 1)
#
# print(f"Our algorithm: {LB}, with an upperbound of {UB}")
# print("\t", X_int)