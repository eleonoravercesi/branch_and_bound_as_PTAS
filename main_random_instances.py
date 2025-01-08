import os
from parse_files import parse_job_scheduling
from BeB.job_scheduling_identical_machines import BeB_JS_ID
import pandas as pd
from exact_models.job_scheduling import identical_machines_job_scheduling
from tqdm import tqdm
import numpy as np
from itertools import product

test_problem = "job_scheduling_identical"
test_type = "random_instances"
config = {'identical' : True, 'uniform' : [], 'timelimit' : 600, 'n_instances' : 30, 'n_jobs' : [50, 100, 500, 1000], 'n_machines' : [2, 5, 10, 20, 50, 100]}

# Create a pandas data frame to store the results
df = pd.DataFrame(columns=["seed", "n_machines", "n_jobs", "epsilon", "lower_bound_type", "branching_rule", "node_selection", "rounding_rule", "makespan", "runtime", "depth", "best_solution", "optimal", "optimality_gap"])


timelimit = config['timelimit']
timelimit_exact = timelimit
n_instances = config['n_instances']
print("Set a time limit of ", timelimit, " seconds")


for n_jobs in config['n_jobs']:
    for n_machines in config['n_machines']:
        # If I have more jobs than machines.....
        if n_jobs > n_machines:
            for seed in range(n_instances):
                print("Solving ", n_jobs, "jobs and ", n_machines, "machines with seed", seed, "exactly ...")
                np.random.seed(seed)
                P = np.random.randint(1, 100, (n_jobs,)) # Random processing times
                # Solve with exact method
                T_opt, X, status = identical_machines_job_scheduling(P, n_machines, timelimit=timelimit_exact)
                tests = product([0.1, 0.01, 0.001], ["linear_relaxation", "binary_search"],
                                ["largest_fractional_job", "largest_fraction"],
                                ["largest_lower_bound", "depth_first", "breadth_first"], ["arbitrary_rounding"])
                tests = list(tests)
                print("Solving ", n_jobs, "jobs and ", n_machines, "machines with seed", seed, "with B&B ...")
                print(f"\tRunning {len(tests)} tests...")
                #for test in tqdm(tests):
                for test in tests:
                    epsilon, lower_bound_type, branching_rule, node_selection, rounding_rule = test
                    # Solve with our branch and bound
                    beb = BeB_JS_ID(P, n_machines, timelimit=config['timelimit'], epsilon=epsilon, lower_bound_type=lower_bound_type,
                                    branching_rule=branching_rule, node_selection=node_selection,
                                    rounding_rule=rounding_rule, verbose=0)
                    T, X, runtime, depth = beb.solve()
                    # Update the pandas dataframe
                    df = df._append({"seed": seed,  "n_machines": n_machines, "n_jobs": len(P),
                                    "epsilon": epsilon, "lower_bound_type": lower_bound_type,
                                    "branching_rule": branching_rule, "node_selection": node_selection,
                                    "rounding_rule": rounding_rule, "makespan": T, "runtime": runtime,
                                    "depth": depth, "best_solution": T_opt, "optimal": status,
                                    "optimality_gap": (T - T_opt) / T_opt}, ignore_index=True)

                    df.to_csv(f"./output/results_{test_problem}_{test_type}.csv", index=False)
                print("\tDone with ", n_jobs, "jobs and ", n_machines, "machines with seed", seed)