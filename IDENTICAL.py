'''
This is the code you want to run to get the experiments for the IDENTICAL benchmark.
It runs tests using several algorithms
(i) Standard (plain) Gurobi
(ii) Gurobi with a gap
(iii) Gurobi with all the preprocessing and cutting planes disabled
(iv) Our B&B with NO grouping
(v) Our B&B with profiling
All the tests are done with a timelimit of 10 minutes
'''
import sys
import os
from parse_files import parse_instance
from algorithms import JS_ILP, JS_LP, JS_LB_BS_identical, BeB_standard, BeB_with_profile
import gurobipy as gp
import time
from math import floor


# Parse the arguments from the command line
try:
    test = sys.argv[1]
    test = bool(int(test)) # 1 = True, 0 = False
except:
    test = False
if not test:
    print("Running in test mode")
    if len(sys.argv) > 1:
        dataset = sys.argv[2]
        epsilon = float(sys.argv[3])
        timelimit = int(sys.argv[4])


if test:
    print("Running in test mode")
    epsilon = 0.1
    dataset = "instancias1a100"
    timelimit = 5

directory_name = "./data/{}".format(dataset)

instances = os.listdir(directory_name)

# Sort by name
instances.sort()

# If we are running in test mode....
if test:
    instances = [instances[0]]

f = open("./output/identical_profiles_{}_{}.csv".format(dataset, epsilon), "w+")
# For redibily, build the header piece by piece
header = "instance_name,n_jobs,n_machines,linear_lb,bs_lb,"
# Standard Gurobi
header += "gurobi_best,gurobi_nodes,gurobi_time,optimal,gurobi_gap," # <-- Last vale is if it is optimal or not
# Gurobi with Gap
header += "gurobi_gap_best,gurobi_gap_nodes,gurobi_gap_time,gurobi_gap_gap,"
# Gurobi with no preprocessing
header += "gurobi_no_preprocessing_best,gurobi_no_preprocessing_nodes,gurobi_no_preprocessing_time,gurobi_no_preprocessing_gap,"
# Our B&B with no grouping
header += "our_best,our_nodes,our_time,our_depth,our_gap,"
# Our B&B with profiling
header += "our_profiling_best,our_profiling_nodes,our_profiling_time,our_profiling_depth,our_profiling_gap\n"
f.write(header)
f.close()

for instance in instances:
    print("Starting instance ", instance, flush=True)
    P = parse_instance(instance, directory_name)
    n_jobs, n_machines = P.shape
    # Make the P a column vector -- we are in the identical case
    P = P[:, 0].reshape(-1, 1)
    to_write = instance + "," + str(n_jobs) + "," + str(n_machines) + ","


    #####################
    # Linear relaxation #
    #####################
    start = time.time()
    lb, _ = JS_LP(P, n_machines=n_machines)
    to_write += str(lb) + ","
    print("\tLinear relaxation: ", round(time.time() - start, 2), "s", flush=True)

    ##################
    # BS Lower Bound #
    ##################
    start = time.time()
    lb, X, _ = JS_LB_BS_identical(P,  n_machines = n_machines, tol = 1e-5)
    to_write += str(lb) + ","
    print("\tBS Lower Bound with Binary search: ", round(time.time() - start, 2), "s", flush=True)

    ###################
    # Standard Gurobi #
    ###################
    start = time.time()
    opt , _ , bb_nodes, runtime, gap, is_opt = JS_ILP(P, n_machines=n_machines, timelimit=timelimit)
    to_write += str(opt) + "," + str(bb_nodes) + "," + str(runtime) + "," + str(is_opt) + "," + str(gap) + ","
    print("\tStandard Gurobi: ", round(time.time() - start, 2), "s", flush=True)

    ###################
    # Gurobi with Gap #
    ###################
    start = time.time()
    opt , _ , bb_nodes, runtime, gap, _ = JS_ILP(P, n_machines=n_machines, timelimit=timelimit, gap = epsilon)
    to_write += str(opt) + "," + str(bb_nodes) + "," + str(runtime) + "," + str(gap) + ","
    print("\tGurobi with Gap: ", round(time.time() - start, 2), "s", flush=True)

    ################################
    # Gurobi with no preprocessing #
    ################################
    start = time.time()
    opt , _ , bb_nodes, runtime, gap, _ = JS_ILP(P, n_machines=n_machines, timelimit=timelimit, presolve = False, cuts = False, gap = epsilon)
    to_write += str(opt) + "," + str(bb_nodes) + "," + str(runtime) + "," + str(gap) + ","
    print("\tGurobi with no preprocessing and no cuts: ", round(time.time() - start, 2), "s", flush=True)

    ########################################
    # Our B&B with no grouping / profiling #
    ########################################
    start = time.time()
    best_objective, best_lb, best_solution, nodes_explored, depth, runtime, optimal  = BeB_standard(P, epsilon, n_machines, timelimit=timelimit)
    gap  = (best_objective - best_lb) / best_objective
    to_write += str(best_objective) + "," + str(nodes_explored) + "," + str(runtime) + "," + str(depth) + "," + str(gap) + ","
    print("\tOur B&B: ", round(time.time() - start, 2), "s", flush=True)

    ##########################
    # Our B&B with profiling #
    ##########################
    start = time.time()
    # Prerocessing first
    P_red = P.copy()
    denominator = sum(P_red) / n_machines
    P_red = P_red / denominator

    # If it exist a job that is smaller than epsilon / n_machines
    if sum(P_red[j] < (epsilon / n_machines) for j in range(n_jobs)) > 1:
        # Create a model. The variables are x_{ij}: group
        r = gp.Model("grouping")
        n_new_vars_max = int(n_machines ** 2 // epsilon)
        X = r.addVars(n_jobs, n_new_vars_max, vtype=gp.GRB.BINARY, name="X")
        # Add the constraints
        for j in range(n_jobs):
            r.addConstr(gp.quicksum(X[i, j] * P_red[i] for i in range(n_jobs)) >= (epsilon / n_machines))
        r.optimize()

        X_rec = dict(zip(range(n_new_vars_max), [] * n_new_vars_max))
        for j in range(n_new_vars_max):
            for i in range():
                if X[i, j].x > 0:
                    X_rec[j].append(i)

        del X # Free the memory

        # Now we have the grouping! Get it
        P_new = []
        for j in range(n_new_vars_max):
            p_new = sum(P[i] * X[i] for i in range(n_jobs) if )
            if p_new > 0:
                P_new.append(p_new)

        print("Finally, we grouped {} jobs".format(len(P_new)))


    # TODO Round down by epsilon / k where k is the exact number of jobs
    value_max = int(max(P) // (epsilon/k))

    # P_flattened = []
    # for p in P:
    #     for i in range(0, value_max + 1):
    #         if i * (epsilon/k) <= p < (i + 1) * (epsilon/k):
    #             P_flattened.append(i)
    #
    # P = np.array(P_flattened).reshape(-1, 1)
    # best_objective, X_best_lb, best_solution, nodes_explored, depth, runtime, optimal = BeB_with_profile(P, epsilon, n_machines, timelimit=timelimit, verbose=False)
    #
    # # Step 1: recover the original solution
    # T_list = []
    # for j in range(n_machines):
    #     T_list.append(np.dot(P_large.T, best_solution[:, j])[0])
    # best_objective = max(T_list)
    #
    # # Step 2: recover the original lb
    # T_list = []
    # for j in range(n_machines):
    #     T_list.append(np.dot(P_large.T, X_best_lb[:, j])[0])
    # best_lb = max(T_list)
    #
    # # Step 3: compute the gap
    # gap = (best_objective - best_lb) / best_objective
    #
    # to_write += str(best_objective) + "," + str(nodes_explored) + "," + str(runtime) + "," + str(depth) + "," + str(gap) + "\n"
    #
    # print("\tOur B&B with profiling: ", round(time.time() - start, 2), "s", flush=True)
    #
    # f = open("./output/identical_profiles_{}_{}.csv".format(dataset, epsilon), "a")
    # f.write(to_write)
    # f.close()
    #
    # print("Finished instance\n*************************************", instance, " in ", round(time.time() - start, 2), "s", flush=True)