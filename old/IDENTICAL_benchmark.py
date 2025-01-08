import os
import time
from old.parse_files import parse_instance
from math import ceil
from algorithms import JS_ILP, JS_LP, JS_LB_BS_identical, BeB_standard


# Pick a directory
bench = "instancias1a100"
directory_name = "./data/{}".format(bench)

instances = os.listdir(directory_name)

# Sort by name
instances.sort()

test = True

if test:
    instances = [instances[0]]


# Open a file to save infos
epsilon = 0.1 # Now, just check it's correct
f = open("./output/identical_{}_{}.csv".format(bench, epsilon), "w+")
f.write(
    "instance_name,n_jobs,n_machines,gurobi_best,gurobi_nodes,gurobi_time,lb_linear_relaxation,gurobi_beb_best,gurobi_beb_nodes,gurobi_beb_time,our_LB,our_best,our_nodes,our_time,our_depth\n")
f.close()

# Sort the instances by number of machines

for instance in instances:
    print("Starting instance ", instance)
    P = parse_instance(instance, directory_name)
    n_jobs, n_machines = P.shape
    # Make the P a column vector -- we are in the identical case
    P = P[:, 0].reshape(-1, 1)

    '''
    Optimal
    '''
    start_OPT = time.time()
    T_OPT, X_OPT, bb_nodes_exact, is_done = JS_ILP(P, n_machines=n_machines, timelimit=10*60) # 10 minutes
    time_OPT = time.time() - start_OPT
    print("\tOptimal value is ", T_OPT, "in ", time.time() - start_OPT, "seconds")
    if not is_done:
        print("\t\tThe ILP did not finish in time")
    ILP_runtime = time.time() - start_OPT
    print("-------------")

    '''
    LB
    '''
    T_LB_gurobi, _ = JS_LP(P, n_machines=n_machines)
    T_LB, _, _ = JS_LB_BS_identical(P, n_machines=n_machines)


    '''
    Gurobi B&B
    '''
    start_gurobi_beb = time.time()
    T_gurobi_beb, _, bb_nodes_gurobi_beb, is_done_gurobi_beb = JS_ILP(P, n_machines=n_machines, verbose = False, presolve = False, cuts = False, gap = epsilon, timelimit = 60*10)  # 10 minutes
    time_gurobi_beb = time.time() - start_OPT
    print("Best value with naive B&B for Gurobi", T_gurobi_beb, "in ", time_gurobi_beb, "seconds")
    if not is_done:
        print("\tGurobi B&B did not finish in time")

    print("-------------")


    '''
    Our B&B
    '''
    best_objective, best_solution, nodes_explored, depth, runtime_total = BeB_standard(P, epsilon, n_machines, verbose = False)
    print("Best solution found our method: ", best_objective)
    print("\tNodes explored: ", nodes_explored)
    if runtime_total > 10*60 + 2:
        print("\tTime limit exceeded")
    print("Time: ", runtime_total)

    # Is the number of nodes explored consistent with the theoretical bound?
    th_bound = ceil((n_machines / epsilon) ** n_machines)
    print("Explored {} nodes out of {}".format(nodes_explored, th_bound))
    # Wrote the results
    f = open("./output/identical_{}_{}.csv".format(bench, epsilon), "a+")
    #f.write(
    #    "instance_name, n_jobs, n_machines, gurobi_best, gurobi_nodes, gurobi_time, lb_linear_relaxation, gurobi_beb_best, gurobi_beb_nodes, gurobi_beb_time, our_LB, our_best, our_nodes, our_time, our_depth\n")
    f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
        instance, n_jobs, n_machines, T_OPT, bb_nodes_exact, ILP_runtime, T_LB_gurobi, T_gurobi_beb, bb_nodes_gurobi_beb, time_gurobi_beb, T_LB,
        best_objective, nodes_explored, runtime_total, depth))
    f.close()

    print("Finished instance ", instance)
