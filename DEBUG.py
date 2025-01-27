import pandas as pd
import numpy as np
from math import ceil
from exact_models.unrelated_job_scheduling import solve_unrelated_job_scheduling
from BeB.unrelated_job_scheduling import BranchAndBound
from itertools import product


jobs_machines_list = [(5, 2)]
node_selection_strategy_list = ["lowest_lower_bound", "depth_first", "breadth_first"]
lower_bound_list = ["lin_relax", "bin_search"]
branching_rule_list = ["max_min_proc", "max_avg_proc"]
rounding_rule_list = ["best_matching", "all_to_shortest"]
epsilon_list = [0.5, 0.1, 0.05, 0.01]

tests_to_do = product(epsilon_list, node_selection_strategy_list, lower_bound_list,
                      branching_rule_list, rounding_rule_list)
tests_to_do = list(tests_to_do)

seed_min = 0
seed_max = 29

# Set up the things you want to record
test_problem = "unrelated_job_scheduling"
test_type = "random_instances"

# for n_jobs, n_machines in jobs_machines_list:
#     print(f"Starting with {n_jobs} - {n_machines}", flush=True)
#     for seed in range(seed_min, seed_max + 1):
#         # Set the seed
#         np.random.seed(seed)
#
#         # Define processing times
#         processing_times = np.random.randint(1, 20, (n_jobs, n_machines)).tolist()
#         print(processing_times)
#
#         OPT_exact, _, status, runtime = solve_unrelated_job_scheduling(processing_times, verbose=2)
#
#         for epsilon, node_selection_strategy, lower_bound, branching_rule, rounding_rule in tests_to_do:
#             print("Doing", epsilon, node_selection_strategy, lower_bound, branching_rule, rounding_rule)
#             # TODO from here onward
#             beb = BranchAndBound(node_selection_strategy, lower_bound, branching_rule, rounding_rule, epsilon)
#             # self.GLB, self.GLB_argmin, self.GUB, time.time() - start, nodes_explored, left_turns, max_depth,  True
#             best_solution, X_int, LB, runtime, nodes_explored, nodes_opt, max_depth, terminate = (
#                 beb.solve(processing_times, verbose=2, opt=OPT_exact))
#
#             print(OPT_exact, best_solution)
#             # assert round(best_solution) <= round(OPT_exact), "Our solution cannot be better than the optimal"
#
#         # Logging
#         print(f"Done with seed {seed}", flush=True)
#

n_jobs, n_machines = 5, 2
seed = 0
processing_times = [[45, 48], [65, 68], [68, 10], [84, 22], [37, 88]]
beb = BranchAndBound("lowest_lower_bound", "bin_search", "max_min_proc", "best_matching", 0.01)
OPT_exact, _, status, runtime = solve_unrelated_job_scheduling(processing_times, verbose=2)
best_solution, X_int, LB, runtime, nodes_explored, nodes_opt, max_depth, terminate = (
    beb.solve(processing_times, verbose=2, opt=OPT_exact))
print(OPT_exact, best_solution)
