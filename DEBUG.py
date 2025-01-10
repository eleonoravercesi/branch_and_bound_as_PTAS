'''
Experiments for the multidimensional knapsack
'''
import numpy as np
from BeB.multi_knapsack import BranchAndBound
from exact_models.multi_knapsack import solve_multi_knapsack
from bounds.multi_knapsack import dantzig_upper_bound, dantzig_upper_bound_linear_relaxation


#
# # Test instance done by hand as well
# n_knapsacks = 2
# n_items = 5
#
# profits = [2, 4,9,12,10]
# weights = [1, 2,3,4,5]
# capacities = [5,6]
#
# alpha = 1


'''

seed    |   knapsack  | items | status  |   alpha   |
-----------------------------------------------------
43          2           10      OK          1
16463       10          20      OK          1
4164        3           10      OK          1
1-42        3           15      OK          1
43          3           15      OK          1
44 - 100    3           15      OK          1
43636       5           20      OK          0.9
43636       5           20      OK          0.95
6106        10          50      OK          0.95
6106        10          50      OK          0.98
41655652    10          50      OK          0.98
41655652    10          50      --          0.99
2646515626  20          100     --         0.90
'''



seed_min = 2646515626
seed_max = 2646515626
alpha = 0.90

for seed in range(seed_min, seed_max + 1):
    print(f"SEED {seed}")
    np.random.seed(seed)
    n_knapsack = 20
    n_items = 100
    profits = np.random.randint(1, 10, (n_items,)).tolist()
    weights = np.random.randint(1, 10, (n_items,)).tolist()

    capacities = np.random.randint(max(weights), 2*max(weights), (n_knapsack,)).tolist()


    total_profit, assignment, status, runtime = solve_multi_knapsack(profits.copy(), weights.copy(), capacities.copy())
    print(f"Total profit (exact): {total_profit}")

beb = BranchAndBound("greatest_upper_bound", "dantzig_upper_bound", "critical_element", "martello_toth_rule", alpha)
LB, X_int, UB, runtime = beb.solve(profits.copy(), weights.copy(), capacities.copy(), verbose = 0)

print(f"Our algorithm: {LB}, with an upperbound of {UB}")
print("\t", X_int)
print("Runtime heuristic: ", {runtime})
#assert abs(LB - total_profit) <= 1e-6