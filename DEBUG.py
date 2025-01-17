import pandas as pd
import numpy as np
from math import ceil
from exact_models.multi_knapsack import SCIP

items_knapsack_list = [(50, 5)]
#seed_min = 5016165 # 0 is trouble, let's try 5016165
seed_min = 0
seed_max = seed_min
tests_to_do = [(0.97, "breadth_first", "kolasar_rule")]



# Set up the things you want to record
test_problem = "multiknapsack"
test_type = "random_instances"

# Create a pandas data frame to store the results
df = pd.DataFrame(columns=["seed", "n_knapsacks", "n_items", "alpha", "branching_rule", "node_selection",
                           "best_solution", "runtime", "depth", "number_of_left_turns", "nodes_explored", "terminate", "optimal_solution"])


for n_items, n_knapsacks in items_knapsack_list:
    print(f"Starting with {n_items} - {n_knapsacks}", flush=True)
    for seed in range(seed_min, seed_max + 1):
        # Set the seed
        np.random.seed(seed)

        # Define profits and weights
        profits = np.random.randint(1, 20, (n_items, )).tolist()
        weights = np.random.randint(1, 20, (n_items, )).tolist()

        print(profits)
        print(weights)

        # Sort the items once for all
        # Step 2: sort the item
        sorted_items = {}
        for j in range(n_items):
            sorted_items[j] = profits[j] / weights[j]

        sorted_items = dict(sorted(sorted_items.items(), key=lambda item: item[1], reverse=True))

        sorted_items = list(sorted_items)

        profits = [profits[j] for j in sorted_items]
        weights = [weights[j] for j in sorted_items]

        # Define capacities
        c_min = min(weights)
        w_sum = sum(weights)
        c_max = ceil(w_sum / n_knapsacks) - c_min # Half of the items can fit in on average


        capacities = np.random.randint(c_min, c_max, (n_knapsacks,)).tolist() # This is just to ensure feasibility
        print(capacities)

        out = SCIP(profits.copy(), weights.copy(), capacities.copy())
