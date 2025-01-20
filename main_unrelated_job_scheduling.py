from itertools import product
import pandas as pd
import numpy as np
from exact_models.unrelated_job_scheduling import solve_unrelated_job_scheduling
from BeB.unrelated_job_scheduling import BranchAndBound

job_machines_list = [(5, 2), (10, 2), (10, 5), (50, 2), (50, 5), (50, 10), (50, 20), (100, 2), (100, 5), (100, 10)]

epsilon_list = [0.5, 0.1, 0.05, 0.01]
node_selection_strategy_list = []
branching_rule_list = []

tests_to_do = product(epsilon_list, node_selection_strategy_list, branching_rule_list)
tests_to_do = list(tests_to_do)

seed_min = 0
seed_max = 29


# Set up the things you want to record
test_problem = "unrelated_job_scheduling"
test_type = "random_instances"

# TODO anything else
# Create a pandas data frame to store the results
df = pd.DataFrame(columns=["seed", "n_knapsacks", "n_items", "alpha", "branching_rule", "node_selection",
                           "best_solution", "runtime", "depth", "nodes_explored", "terminate", "number_of_nodes_for_optimality", "optimal_solution", "opt_gap"])

def opt_gap(best_solution, OPT_exact, tol = 1e-6):
    return abs(best_solution - OPT_exact) / max(tol, OPT_exact, best_solution)


for n_items, n_knapsacks in job_machines_list:
    print(f"Starting with {n_items} - {n_knapsacks}", flush=True)
    for seed in range(seed_min, seed_max + 1):
        # Set the seed
        np.random.seed(seed)

        # Define the completion times
        completion_times = np.random.randint(1, 100, (n_items, n_knapsacks)).tolist()


        OPT_exact, _, status, runtime = solve_unrelated_job_scheduling(completion_times, verbose=0)

        for epsilon_list, node_selection_strategy, branching_rule in tests_to_do:
            print("Doing", epsilon_list, node_selection_strategy, branching_rule)
            # TODO from here onward
            beb = BranchAndBound(node_selection_strategy, "dantzig_upper_bound", branching_rule, "martello_toth_rule", alpha)
            # self.GLB, self.GLB_argmin, self.GUB, time.time() - start, nodes_explored, left_turns, max_depth,  True
            best_solution, X_int, UB, runtime, nodes_explored, opt_node, left_turns, max_depth, terminate = beb.solve(profits.copy(), weights.copy(), capacities.copy(), opt=OPT_exact, verbose=0)


            print(OPT_exact, best_solution)
            assert  round(best_solution) <= round(OPT_exact)

            # Logging
            print(f"Done with seed {seed}", flush=True)

            df = df._append(dict(seed=seed, n_knapsacks=n_knapsacks, n_items=n_items, alpha=alpha, branching_rule=branching_rule,
                                 node_selection=node_selection_strategy, best_solution=best_solution, runtime=runtime,
                                 depth=max_depth, number_of_left_turns=left_turns, nodes_explored=nodes_explored, terminate=terminate, number_of_nodes_for_optimality=opt_node,
                                 optimal_solution=OPT_exact, opt_gap = opt_gap(best_solution, OPT_exact)), ignore_index=True)

            df.to_csv(f"./output/results_{test_problem}_{test_type}.csv", index=False)