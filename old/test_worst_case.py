import heapq
import time

import numpy as np
from networkx.classes import number_of_nodes, nodes

from algorithms import JS_ILP, JS_LP, JS_LB_BS_fast_idendical, JS_LB_BS
from utils import *
from math import ceil

# Sort the instances by number of machines
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
    elif k == 6:
        P = [1,1,2,3,5,8,13]
        n_machines = 2
        epsilon = 1/3
    if len(P) == 0:
        raise ValueError("Invalid setting")
    return P, n_machines, epsilon

P, n_machines, epsilon = setting(6)
# Make the P a column vector -- we are in the identical case
P = np.asarray(P).reshape(-1, 1)

seed_debug = 1
for seed in range(seed_debug, seed_debug + 1):
    print(seed)
    np.random.seed(seed)
    P = np.random.randint(1, 100, size=(10, 1))
    n_machines = 2
    epsilon = 1/3
    '''
    Optimal
    '''
    start_OPT = time.time()
    T_OPT, X_OPT, bb_nodes_exact, is_done = JS_ILP(P, n_machines=n_machines, timelimit=10*60) # 10 minutes
    time_OPT = time.time() - start_OPT
    print("Optimal value is ", T_OPT, "in ", time.time() - start_OPT, "seconds")
    if not is_done:
        print("\tThe ILP did not finish in time")
    ILP_runtime = time.time() - start_OPT

    '''
    LB
    '''
    T_LB_gurobi, _ = JS_LP(P, n_machines=n_machines)


    '''
    Gurobi B&B
    '''
    start_gurobi_beb = time.time()
    T_gurobi_beb, _, bb_nodes_gurobi_beb, is_done_gurobi_beb = JS_ILP(P, n_machines=n_machines, verbose = False, presolve = False, cuts = False, gap = epsilon, timelimit = 60*10)  # 10 minutes
    time_gurobi_beb = time.time() - start_OPT
    print("Best value with naive B&B for Gurobi", T_gurobi_beb, "in ", time_gurobi_beb, "seconds")
    if not is_done:
        print("\tGurobi B&B did not finish in time")


    '''
    Our B&B
    '''
    start = time.time()
    T_LB, X_LB = JS_LB_BS_fast_idendical(P, n_machines=n_machines)
    print("Time for the LB: ", time.time() - start)
    X_best = round_LP_solution_matching(X_LB, P, n_machines=n_machines)
    T_max_for_each_machine = []
    for j in range(n_machines):
        T_max_for_each_machine.append(np.dot(P.T, X_best[:, j])[0])
    T_best = max(T_max_for_each_machine)
    print("Starting with a best solution of ", T_best, "and the Gurobi best value is ", T_OPT)

    verbose = True

    # Priority queue for B&B nodes (max-heap based on lower bound, that's why we use -T_LB)
    pq = []

    # Start with the root node
    heapq.heappush(pq, (-T_LB, [], X_LB))

    best_solution = X_best
    best_objective = T_best
    best_lower_bound = T_LB

    nodes_explored = 0

    while pq:
        # Get the node with the lowest bound
        current_lb, fixed_vars, current_solution = heapq.heappop(pq)

        # Round the current solution
        X_rounded = round_LP_solution_matching(current_solution, P, n_machines=n_machines)
        T_max_for_each_machine = []
        for j in range(n_machines):
            T_max_for_each_machine.append(np.dot(P.T, X_best[:, j])[0])
        T_rounded = max(T_max_for_each_machine)

        nodes_explored += 1

        # Change the sign for your convenience
        current_lb = -current_lb

        # Prune if the current lower bound is worse than the best found so far
        if current_lb >= best_objective:
            continue

        # Check if the current solution is feasible and better than the best
        if is_integer_sol(current_solution):
            current_objective = current_lb
            if current_objective < best_objective:
                best_solution = current_solution
                best_objective = current_objective
                print("New best solution found: ", best_objective, "in ", time.time() - start,
                      "seconds, nodes explored: ", nodes_explored)
            continue

        # Branch on the variable that has the largest job
        i = find_largest_fractional(current_solution, P)
        for j in range(n_machines):
            new_fixed_vars = fixed_vars + [((i, j), 1)]
            T_new, X_new = JS_LB_BS(P, n_machines=n_machines, fixed=new_fixed_vars)

            if T_new < best_objective:
                heapq.heappush(pq, (-T_new, new_fixed_vars, X_new))

        # Exit control: Pick the smallest LB among the active nodes
        if pq:
            best_lb = -pq[0][0]

        if best_objective / best_lb <= 1 + epsilon:
            print("Optimality gap is within the tolerance!")
            break

        if nodes_explored % 1000 == 0:
            print("Nodes explored: ", nodes_explored)
            print("Best solution found: ", best_objective)

    if nodes_explored >= 10:
        print("Too many nodes explored")
        break
    print("Best solution found: ", best_objective)
    print("Optimal value is ", T_OPT)
    print("Nodes explored: ", nodes_explored)
    runtime_total = time.time() - start
    depth = len(fixed_vars)
    print("Time: ", runtime_total)

    print("-----------------")

