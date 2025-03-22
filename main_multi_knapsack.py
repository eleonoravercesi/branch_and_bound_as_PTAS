import  numpy as np
from exact_models.multi_knapsack import solve_multi_knapsack
from BeB.multi_knapsack import BranchAndBound
from itertools import product
from math import ceil
import pandas as pd

items_knapsack_list = [(5, 2), (10, 2), (10, 5), (50, 2), (50, 5), (50, 10), (50, 15), (100, 2), (100, 5), (100, 10), (100, 15)]

alpha_list = [0.5, 0.8, 0.95, 0.97]
node_selection_strategy_list = ["greatest_upper_bound", "depth_first", "breadth_first"]
branching_rule_list = ["critical_element", "profit_per_weight_ratio", "kolasar_rule"]

tests_to_do = product(alpha_list, node_selection_strategy_list, branching_rule_list)
tests_to_do = list(tests_to_do)

seed_min = 0
seed_max = 29

'''
Single test
'''
# items_knapsack_list = [(50, 5)]
# seed_min = 0
# seed_max = seed_min
# tests_to_do = [(0.97, "breadth_first", "kolasar_rule")]



# Set up the things you want to record
test_problem = "multiknapsack"
test_type = "random_instances"

# Create a pandas data frame to store the results
df = pd.DataFrame(columns=["seed", "n_knapsacks", "n_items", "alpha", "branching_rule", "node_selection",
                           "best_solution", "best_bound", "runtime", "depth", "number_of_left_turns", "nodes_explored", "terminate", "number_of_nodes_for_optimality", "optimal_solution", "opt_gap"])

def opt_gap(best_solution, OPT_exact, tol = 1e-6):
    return abs(best_solution - OPT_exact) / max(tol, OPT_exact, best_solution)


for n_items, n_knapsacks in items_knapsack_list[10:]:
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


        OPT_exact, _, status, runtime = solve_multi_knapsack(profits.copy(), weights.copy(), capacities.copy())

        for alpha, node_selection_strategy, branching_rule in tests_to_do:
            print("Doing", alpha, node_selection_strategy, branching_rule)
            beb = BranchAndBound(node_selection_strategy, "dantzig_upper_bound", branching_rule, "martello_toth_rule", alpha)
            # self.GLB, self.GLB_argmin, self.GUB, time.time() - start, nodes_explored, left_turns, max_depth,  True
            best_solution, X_int, UB, runtime, nodes_explored, opt_node, left_turns, max_depth, terminate = beb.solve(profits.copy(), weights.copy(), capacities.copy(), opt=OPT_exact, verbose=0)


            print(OPT_exact, best_solution)
            assert  round(best_solution) <= round(OPT_exact)

            # Logging
            print(f"Done with seed {seed}", flush=True)

            df = df._append(dict(seed=seed, n_knapsacks=n_knapsacks, n_items=n_items, alpha=alpha, branching_rule=branching_rule,
                                 node_selection=node_selection_strategy, best_solution=best_solution, best_bound=UB, runtime=runtime,
                                 depth=max_depth, number_of_left_turns=left_turns, nodes_explored=nodes_explored, terminate=terminate, number_of_nodes_for_optimality=opt_node,
                                 optimal_solution=OPT_exact, opt_gap = opt_gap(best_solution, OPT_exact)), ignore_index=True)

        if seed + 1 % 5 == 0:
            # Save it every 5 seeds
            df.to_csv(f"./output/results_{test_problem}_{test_type}.csv", index=False)