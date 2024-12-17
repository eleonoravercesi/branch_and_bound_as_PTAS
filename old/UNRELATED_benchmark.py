import os
import heapq
import time
from algorithms import JS_ILP, JS_LP, JS_LB_BS_fast_idendical
from parse_files  import parse_instance
from utils import *
from math import ceil
import platform

# Pick a directory
bench = "instancias1a100"
directory_name = "./data/{}".format(bench)

instances = os.listdir(directory_name)

# Sort by name
instances.sort()

# If it is on my Mac, instances has just length 3
test = True
if test:
    instances = [instances[0]]


# Open a file to save infos
epsilon = 0.1
f = open("unrelated_{}_{}.csv".format(bench, epsilon), "w+")
f.write(
    "instance_name,n_jobs,n_machines,gurobi_best,gurobi_nodes,gurobi_time,lb_linear_relaxation,gurobi_beb_best,gurobi_beb_nodes,gurobi_beb_time,our_LB,our_best,our_nodes,our_time,our_depth\n")
f.close()

# Sort the instances by number of machines

for instance in instances:
    print("Starting instance ", instance)
    P = parse_instance(instance, directory_name)
    n_jobs, n_machines = P.shape

    '''
    Optimal
    '''
    start_OPT = time.time()
    T_OPT, X_OPT, bb_nodes_exact, is_done = JS_ILP(P,  timelimit=10*60) # 10 minutes
    time_OPT = time.time() - start_OPT
    print("Optimal value is ", T_OPT, "in ", time.time() - start_OPT, "seconds")
    if not is_done:
        print("\tThe ILP did not finish in time")
    ILP_runtime = time.time() - start_OPT

    '''
    LB
    '''
    T_LB_gurobi, _ = JS_LP(P)


    '''
    Gurobi B&B
    '''
    start_gurobi_beb = time.time()
    T_gurobi_beb, _, bb_nodes_gurobi_beb, is_done_gurobi_beb = JS_ILP(P, verbose = False, presolve = False, cuts = False, gap = epsilon, timelimit = 60*10)  # 10 minutes
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

    start_beb = time.time()
    while pq and time.time() - start_beb < 10*60:  # 10 minutes
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
            T_new, X_new = JS_LB_BS_fast_idendical(P, n_machines=n_machines, fixed=new_fixed_vars)

            if T_new < best_objective:
                heapq.heappush(pq, (-T_new, new_fixed_vars, X_new))

        # Exit control: Pick the smallest LB among the active nodes
        if pq:
            best_lb = -pq[0][0]

        if best_objective / best_lb <= 1 + epsilon:
            break

    print("Best solution found: ", best_objective)
    print("Optimal value is ", T_OPT)
    print("Nodes explored: ", nodes_explored)
    runtime_total = time.time() - start
    if runtime_total > 10*60 + 2:
        print("Time limit exceeded")
    depth = len(fixed_vars)
    print("Time: ", runtime_total)

    # Is the number of nodes explored consistent with the theoretical bound?
    th_bound = ceil((n_machines / epsilon) ** n_machines)
    print("Explored {} nodes out of {}".format(nodes_explored, th_bound))
    # Wrote the results
    f = open("identical_{}_{}.csv".format(bench, epsilon), "a+")
    #f.write(
    #    "instance_name, n_jobs, n_machines, gurobi_best, gurobi_nodes, gurobi_time, lb_linear_relaxation, gurobi_beb_best, gurobi_beb_nodes, gurobi_beb_time, our_LB, our_best, our_nodes, our_time, our_depth\n")
    f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
        instance, n_jobs, n_machines, T_OPT, bb_nodes_exact, ILP_runtime, T_LB_gurobi, T_gurobi_beb, bb_nodes_gurobi_beb, time_gurobi_beb, T_LB,
        best_objective, nodes_explored, runtime_total, depth))
    f.close()

    print("Finished instance ", instance)
