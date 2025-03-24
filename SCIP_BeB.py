"Test the SCIP solver on the multi-knapsack and Unrelated Job Scheduling problems."

import numpy as np
from math import ceil
from tqdm import tqdm

# Choose wich test to run
test = "unrelated_job_scheduling"

seed_min = 0
seed_max = 29
times = []

if test == "multi_knapsack":
    from exact_models.multi_knapsack import SCIP

    n_items = 100
    n_knapsacks = 15
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

        OPT_exact, _, status, runtime = SCIP(profits.copy(), weights.copy(), capacities.copy(), node_limit=10**4, verbose=False)

        times.append(runtime)


elif test == "unrelated_job_scheduling":
    from exact_models.unrelated_job_scheduling import SCIP

    n_jobs = 50
    n_machines = 10

    for seed in tqdm(range(seed_min, seed_max + 1)):
        # Set the seed
        np.random.seed(seed)

        # Define the completion times
        processing_times = np.random.randint(1, 100, (n_jobs, n_machines)).tolist()

        OPT_exact, _, status, runtime = SCIP(processing_times, nodes_limit=10**4, verbose=False)

        times.append(runtime)

else:
    raise ValueError("Invalid test")


print("Test:", test)
print("Mean time", np.mean(times))
print("Standard Deviation", np.std(times))