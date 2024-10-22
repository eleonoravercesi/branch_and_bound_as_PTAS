import time
import heapq
import os
import numpy as np
from algorithms import BeB_standard
from utils import round_LP_solution_matching
from parse_files import parse_instance

def list_scheduling_algorithm_identical_with_fixed_jobs(P, n_machines = None, fixed = []):
    '''
     # jobs are arriving ony by one;
    # you assign each of them to the machine where you can start processing them the earliest
    # (e.g. the one whose completion time so far is the smallest)
    :param P:
    :param n_machines:
    :return:
    '''
    n_jobs = P.shape[0]
    if n_machines == None:
        raise ValueError("You must specify the number of machines in this case")
    X = np.zeros((n_jobs, n_machines))
    T = np.zeros(n_machines)

    jobs_assigned = []
    if len(fixed) > 0:
        for (i, j), _ in fixed:
            T[j] += P[i, 0]
            X[i, j] = 1
            jobs_assigned.append(i)

    for i in range(n_jobs):
        if i not in jobs_assigned:
            # Find the machine with the smallest completion time
            j = np.argmin(T)
            X[i, j] = 1
            T[j] += P[i, 0]
    return max(T), X

def new_LB(P, n_machines, fixed = []):
    '''
    This lowerbound is pretty trivial
    :param P:
    :return:
    '''
    # Compute overheads
    T = np.zeros(n_machines)
    X = np.zeros((len(P), n_machines))
    for (i, j) in fixed:
        T[j] += P[i]
        X[i, j] = 1

    # Take the maximum
    T_max = np.max(T)

    # Compute a stupid lowerbound
    LB = sum(P) / n_machines

    # Get the jobs already assigned
    job_assigned = [i for (i, j) in fixed]
    var_to_branch_on = np.argmax(P[[i for i in range(len(P)) if i not in job_assigned]])

    if T_max > LB:
        _, X = list_scheduling_algorithm_identical_with_fixed_jobs(P, n_machines=n_machines, fixed=fixed)
        # Get the index of the max among the unfixed jobs
        return T_max, X, var_to_branch_on
    else:
        X = np.zeros((len(P), n_machines))
        for (i, j), _ in fixed:
            X[i, j] = 1
        j = 0
        all_jobs = [i for i in range(len(P))]
        i = all_jobs.pop(0)
        while len(all_jobs) > 0:
            # TODO must me finished and fixed 
            if i not in job_assigned:
                # Find k such that T[j] + k*P[i] <= LB
                k = (LB - T[j]) / P[i]
                if k > 0:
                    # assign it
                    X[i, j] = k
                    T[j] += k * P[i]
                    i = all_jobs.pop(0)
                else:
                    j += 1
        assert max(T) <= LB
        X = round_LP_solution_matching(X, P, n_machines=n_machines)
        return LB, X, var_to_branch_on

def BeB_new_LB(P, epsilon, n_machines, timelimit = 10*60, verbose = False):
    '''

    :param P:
    :param epsilon:
    :param verbose:
    :return:
    '''
    optimal = False
    depth = 0
    start = time.time()
    T_LB, X_best, var_to_branch_on = new_LB(P, n_machines)
    T_max_for_each_machine = np.zeros(n_machines)
    for i in range(len(P)):
        for j in range(n_machines):
            T_max_for_each_machine[j] += P[i] * X_best[i, j]
    T_best = max(T_max_for_each_machine)

    # Priority queue for B&B nodes (max-heap based on lower bound, that's why we use -T_LB)
    pq = []

    # Start with the root node
    heapq.heappush(pq, (-T_LB, [], var_to_branch_on))

    best_solution = X_best
    best_objective = T_best
    best_lb = T_LB
    nodes_explored = 0

    start_beb = time.time()
    while pq and time.time() - start_beb < timelimit:
        # Get the node with the lowest bound
        current_lb, fixed_vars, var_to_branch_on = heapq.heappop(pq)

        nodes_explored += 1

        # Change the sign for your convenience
        current_lb = -current_lb

        # Prune if the current lower bound is worse than the best found so far
        if current_lb >= best_objective:
            continue

        else:
            for j in range(n_machines):
                new_fixed_vars = fixed_vars + [((var_to_branch_on, j), 1)]
                T_new, X_new, is_optimal = new_LB(P, n_machines=n_machines, fixed=new_fixed_vars)
                if verbose:
                    print("New node: ", new_fixed_vars, " with LB: ", T_new)
                if T_new < best_objective and not is_optimal:
                    # Because if it is optimal, we have the best we can get at that node
                    heapq.heappush(pq, (-T_new, new_fixed_vars, X_new))


        # Exit control: Pick the smallest LB among the active nodes
        if pq:
            best_lb = -pq[0][0]

        if best_objective / best_lb <= 1 + epsilon:
            if verbose:
                print("Finished with best objective: ", best_objective, " and best LB: ", best_lb)
            depth = len(fixed_vars)
            optimal = True
            break

    return best_objective, best_lb, best_solution, nodes_explored, depth, time.time() - start, optimal

test = True
if test:
    print("Running in test mode")
    epsilon = 0.01
    dataset = "instancias1a100"
    timelimit = 5

directory_name = "./data/{}".format(dataset)

instances = os.listdir(directory_name)

# Sort by name
instances.sort()

instances = [instances[0]]

for instance in instances:
    P = parse_instance(instance, directory_name)
    n_jobs, n_machines = P.shape
    # Make the P a column vector -- we are in the identical case
    P = P[:, 0].reshape(-1, 1)

    best_objective, best_lb, best_solution, nodes_explored, depth, runtime, optimal = BeB_standard(P, epsilon,
                                                                                                   n_machines,
                                                                                               timelimit=timelimit)
    gap = (best_objective - best_lb) / best_objective
    print("Old B&B: ", round(runtime, 2), "s", flush=True)
    print("\tBest objective: ", best_objective, flush=True)


    best_objective, best_lb, best_solution, nodes_explored, depth, runtime, optimal = BeB_new_LB(P, epsilon,
                                                                                                   n_machines,
                                                                                                   timelimit=timelimit)
    gap = (best_objective - best_lb) / best_objective
    print("New B&B: ", round(runtime, 2), "s", flush=True)
    print("\tBest objective: ", best_objective, flush=True)
