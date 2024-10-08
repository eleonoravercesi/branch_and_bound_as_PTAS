import numpy as np

from algorithms import JS_LB_BS_fast_idendical, JS_LB_BS
import time



def setting(k):
    P = []
    n_machines = 0
    epsilon = 0
    if k == 1:
        P = 99*[(100 + 1) ** 2] + [1, 1]
        n_machines = 2
        epsilon = 1/100
    elif k == 2: # Example 2 in the paper
        P = 9*[(9 + 2)**2] + [1, 1]
        n_machines = 2
        epsilon = 1/10
    elif k == 3: # Truly changes both
        P = 21 * [2]
        n_machines = 2
        epsilon = 1/100
    elif k == 4: # rtThis to show that out Lb is better
        P = 13 * [1]
        n_machines = 2
        epsilon = 1/100
    elif k == 5: # This is for wors case analysis
        P = 13 * [2]
        n_machines = 2
        epsilon = 1/100
    if len(P) == 0:
        raise ValueError("Invalid setting")
    return P, n_machines, epsilon


if __name__ == "__main__":
    # Pick a directory
    bench = "instancias1a100"
    directory_name = "./data/{}".format(bench)

    # TODO
    #instance = "544.txt"
    #print("Starting instance ", instance)
    #P = parse_instance(instance, directory_name)
    # n_jobs, n_machines = P.shape

    P, n_machines, epsilon = setting(5)
    n_jobs = len(P)
    P = np.array(P).reshape(-1, 1)
    # Make the P a column vector -- we are in the identical case
    #P = P[:, 0].reshape(-1, 1)
    start = time.time()
    out = JS_LB_BS(P, n_machines=n_machines, tol=1e-5)
    print("\tTime elapsed for the non-fast version is ", time.time() - start)
    start_fast = time.time()
    out_fast = JS_LB_BS_fast_idendical(P, n_machines=n_machines, tol=1e-5, verbose=False)
    print("\tTime elapsed for the fast version is ", time.time() - start_fast)
    assert out[0] == out_fast[0]
