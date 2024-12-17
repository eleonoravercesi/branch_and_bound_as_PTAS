import numpy as np

def list_scheduling_algorithm_identical(P, n_machines):
    '''
     # jobs are arriving ony by one;
    # you assign each of them to the machine where you can start processing them the earliest
    # (e.g. the one whose completion time so far is the smallest)
    '''
    n_jobs = len(P)
    X = {}
    T = np.zeros(n_machines)
    for j in range(n_jobs):
        # Find the machine with the smallest completion time
        i = np.argmin(T)
        X[(j, i)] = 1
        T[i] += P[j]
    return max(T), X