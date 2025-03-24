"Test the SCIP solver on the multi-knapsack and Unrelated Job Scheduling problems."
import sys
import numpy as np
from math import ceil
from tqdm import tqdm

from main_multi_knapsack import seed_min

# Pars arguments from command line
test = sys.argv[0]
assert test in ["multi_knapsack", "unrelated_job_scheduling"], "Invalid test, must be either multi_knapsack or unrelated_job_scheduling"

# Define the seed range
seed_min = int(sys.argv[1])
seed_max = int(sys.argv[2])

# Collect the runtimes
times = []


if test == "multi_knapsack":
    from exact_models.multi_knapsack import SCIP

    n_items = int(sys.argv[3])
    n_knapsacks = int(sys.argv[4])

    for seed in tqdm(range(seed_min, seed_max + 1)):
        # Set the seed
        np.random.seed(seed)

        # Define profits and weights
        profits = np.random.randint(1, 20, (n_items, )).tolist()
        weights = np.random.randint(1, 20, (n_items, )).tolist()


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

        # Change the nodes limit to None if you want to solve the problem to optimality
        OPT_exact, _, status, runtime = SCIP(profits.copy(), weights.copy(), capacities.copy(), node_limit=10**4, verbose=False)

        times.append(runtime)


elif test == "unrelated_job_scheduling":
    from exact_models.unrelated_job_scheduling import SCIP

    n_jobs = int(sys.argv[3])
    n_machines = int(sys.argv[4])

    for seed in tqdm(range(seed_min, seed_max + 1)):
        # Set the seed
        np.random.seed(seed)

        # Define the completion times
        processing_times = np.random.randint(1, 100, (n_jobs, n_machines)).tolist()

        # Change the nodes limit to None if you want to solve the problem to optimality
        OPT_exact, _, status, runtime = SCIP(processing_times, nodes_limit=10**4, verbose=False)

        times.append(runtime)

else:
    raise ValueError("Invalid test")


print("Test:", test)
print("Mean time", np.mean(times))
print("Standard Deviation", np.std(times))