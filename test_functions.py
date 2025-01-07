from exact_models.job_scheduling import identical_machines_job_scheduling
from BeB.job_scheduling_identical_machines import BeB_JS_ID
import numpy as np
from utils import is_integer_sol

for seed in range(1, 5):
    np.random.seed(seed)
    print(f"Seed: {np.random.get_state()[1][0]}")
    P = np.random.randint(1, 100, 20).tolist()  # Job processing times
    machines = 2  # Number of identical machines

    # Print the seed for reproducibility

    # makespan, X = linear_relaxation(p, machines)
    #
    # if makespan is not None:
    #     print("\nJob Assignments (Fractional Values):")
    #     for (j, i) in X.keys():
    #         print(f"Job {j} -> Machine {i}: {X[j, i]}")
    #     print(f"\nMakespan: {makespan}")

    # EXACT MODEL
    makespan, X = identical_machines_job_scheduling(P, machines)

    print("Exact Model:")
    if makespan is not None:
        # print("\nJob Assignments:")
        # for (j, i) in X.keys():
        #     print(f"Job {j} -> Machine {i}: {X[j, i]}")
        print(f"\tMakespan with exact method: {makespan}")

    # PTAS
    print("PTAS:")
    beb = BeB_JS_ID(P, machines, lower_bound_type="binary_search", epsilon=0.001, verbose=0)
    T, X, runtime, depth = beb.solve()
    assert is_integer_sol(X), "PTAS solution is not integer" + str(X)
    assert T >= makespan, f"PTAS makespan {T} is smaller than the exact makespan {makespan}"
    # if T is not None:
    #     print("\nJob Assignments:")
    #     for (j, i) in X.keys():
    #         print(f"Job {j} -> Machine {i}: {X[j, i]}")
    print(f"\tMakespan: {T}")
    print("\t Depth --> ", depth)