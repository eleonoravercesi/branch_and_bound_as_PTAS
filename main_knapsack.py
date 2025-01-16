import  numpy as np
from exact_models.multi_knapsack import solve_multi_knapsack
from BeB.multi_knapsack import BranchAndBound
from itertools import product
from math import ceil

# items_knapsack_list = [(5, 2), (10, 2), (10, 5), (50, 2), (50, 5), (50, 10), (100, 2), (100, 5), (100, 10), (100, 20)]
#
# alpha_list = [0.5, 0.8, 0.95, 0.97]
# node_selection_strategy_list = ["greatest_upper_bound", "depth_first", "breadth_first"]
# branching_rule_list = ["critical_element", "profit_per_weight_ratio", "kolasar_rule"]
#
# tests_to_do = product(alpha_list, node_selection_strategy_list, branching_rule_list)
# tests_to_do = list(tests_to_do)
#
# seed_min = 0
# seed_max = 29

items_knapsack_list = [(5, 2)]
seed_min = 2
seed_max = seed_min # Bad seed = 1 with 5, 2
tests_to_do = [(0.97, "breadth_first",  "profit_per_weight_ratio")]

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

        # Define capacities
        w_sum = sum(weights)
        c_max = ceil(n_knapsacks * w_sum / 2)

        capacities = np.random.randint(min(weights), c_max, (n_knapsacks,)).tolist() # This is just to ensure feasibility
        print(capacities)


        OPT_exact, _, status, runtime = solve_multi_knapsack(profits.copy(), weights.copy(), capacities.copy())

        for alpha, node_selection_strategy, branching_rule in tests_to_do:
            print("Doing", alpha, node_selection_strategy, branching_rule)
            beb = BranchAndBound(node_selection_strategy, "dantzig_upper_bound", branching_rule, "martello_toth_rule", alpha)
            # self.GLB, self.GLB_argmin, self.GUB, time.time() - start, nodes_explored, left_turns, max_depth,  True
            best_solution, X_int, UB, runtime, nodes_explored, left_turns, max_depth, terminate = beb.solve(profits.copy(), weights.copy(), capacities.copy(), verbose=2)

        # Logging
        print(f"Done with seed {seed}", flush=True)

        print(OPT_exact, best_solution)