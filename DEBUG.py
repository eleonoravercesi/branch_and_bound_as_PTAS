import numpy as np
from bounds.multi_knapsack import dantzig_upper_bound, dantzig_upper_bound_linear_relaxation

fixed = [(31, 9), (36, 9), (39, 8), (41, 8), (16, 7), (26, 7), (11, 6), (38, 5), (25, 8), (8, 5), (12, 10)]

seed_min = 6106
seed_max = 6106
alpha = 0.99



for seed in range(seed_min, seed_max + 1):
    print(f"SEED {seed}")
    np.random.seed(seed)
    n_knapsack = 10
    n_items = 50
    profits = np.random.randint(1, 10, (n_items,)).tolist()
    weights = np.random.randint(1, 10, (n_items,)).tolist()
    capacities = np.random.randint(max(weights), 2*max(weights), (n_knapsack,)).tolist()

    for (j, i) in fixed:
        if i < n_knapsack:
            capacities[i] -= weights[j]



    X, OPT, _ = dantzig_upper_bound(profits.copy(), weights.copy(), capacities, fixed)
    #X, OPT, _ = dantzig_upper_bound_linear_relaxation(profits.copy(), weights.copy(), capacities, fixed)

    for (j, i) in fixed:
        if i < n_knapsack:
            X[(j, i)] = 1
            OPT += profits[j]

    print(OPT)