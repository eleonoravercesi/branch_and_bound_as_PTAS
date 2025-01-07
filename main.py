import os
from parse_files import parse_job_scheduling
from BeB.job_scheduling_identical_machines import BeB_JS_ID
import pandas as pd
from exact_models.job_scheduling import identical_machines_job_scheduling
from tqdm import tqdm
from itertools import product

test_type = "job_scheduling_identical"
config = {'dir' : './data/instancias1a100', 'identical' : True, 'uniform' : [], 'timelimit' : 1}

# Create a pandas data frame to store the results
df = pd.DataFrame(columns=["dir", "instance", "n_machines", "n_jobs", "epsilon", "lower_bound_type", "branching_rule", "node_selection", "rounding_rule", "makespan", "runtime", "depth", "best_solution", "optimal", "optimality_gap"])

dir = config['dir']
files = os.listdir(dir)
timelimit = config['timelimit']


for file in files:
    P, n_machines = parse_job_scheduling(file, dir, identical=config['identical'], uniform=config['uniform'])
    # Solve with exact method
    T_opt, X, status = identical_machines_job_scheduling(P, n_machines, timelimit=timelimit)
    tests = product([0.1, 0.01, 0.001], ["linear_relaxation", "binary_search"],
                    ["largest_fractional_job", "largest_fraction"],
                    ["largest_lower_bound", "depth_first", "breadth_first"], ["arbitrary_rounding"])
    tests = list(tests)
    print("Solving ", dir, file)
    print(f"\tRunning {len(tests)} tests...")
    for test in tqdm(tests):
        epsilon, lower_bound_type, branching_rule, node_selection, rounding_rule = test
        # Solve with our branch and bound
        beb = BeB_JS_ID(P, n_machines, timelimit=config['timelimit'], epsilon=epsilon, lower_bound_type=lower_bound_type,
                        branching_rule=branching_rule, node_selection=node_selection,
                        rounding_rule=rounding_rule, verbose=0)
        T, X, runtime, depth = beb.solve()
        # Update the pandas dataframe
        df = df._append({"dir": dir, "instance": file, "n_machines": n_machines, "n_jobs": len(P),
                        "epsilon": epsilon, "lower_bound_type": lower_bound_type,
                        "branching_rule": branching_rule, "node_selection": node_selection,
                        "rounding_rule": rounding_rule, "makespan": T, "runtime": runtime,
                        "depth": depth, "best_solution": T_opt, "optimal": status,
                        "optimality_gap": (T - T_opt) / T_opt}, ignore_index=True)

        df.to_csv(f"./output/results_{test_type}.csv", index=False)
    print("\tDone with ", file)
