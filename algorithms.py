import gurobipy as gp
from gurobipy import GRB
import numpy as np
from math import log10
import time
from utils import is_integer_sol, find_largest_fractional, round_LP_solution_matching
import  heapq

def JS_ILP(P, n_machines = None, verbose = False, presolve = True, cuts = True, gap = None, timelimit = None, seed=42):
    if n_machines == None:
        # You want to solve the problem "as it is"
        n_items, n_machines = P.shape
    else:
        # Your machines are identical and you have m of them, hence, just copy the column vector T m times
        try:
            assert P.shape[1] == 1  # T is a column vector
        except AssertionError:
            raise ValueError("T must be a column vector in the case of identical machines")
        # Copy the column vector T m times
        P = np.tile(P, n_machines)
        n_items = P.shape[0]
    m = gp.Model("exact")
    # First, set a seed
    m.setParam('Seed', seed)
    if not verbose:
        m.setParam('OutputFlag', 0)
        m.setParam('LogToConsole', 0)
    if not presolve:
        m.setParam('Presolve', 0)
    if not cuts:
        m.setParam('Cuts', 0)
    if gap != None:
        m.setParam('MIPGap', gap)
    if timelimit != None:
        m.setParam('TimeLimit', timelimit)
    # An integer variable for the makespan
    T = m.addVar(vtype=GRB.CONTINUOUS, name="M")  # Makespan
    # A binary variable x_i_j if variable i is assigned to machine j
    x = m.addVars(n_items, n_machines, vtype=GRB.BINARY, name="x")
    # Constraint: assign each job to exactly one machine
    for i in range(n_items):
        m.addConstr(sum(x[i, j] for j in range(n_machines)) == 1)
    # Constraint: makespan is the maximum completion time
    for j in range(n_machines):
        m.addConstr(sum(x[i, j] * P[i, j] for i in range(n_items)) <= T)
    # Minimize the makespan
    m.setObjective(T, GRB.MINIMIZE)
    m.optimize()
    X = np.zeros((n_items, n_machines))
    for i in range(n_items):
        for j in range(n_machines):
            X[i, j] = x[i, j].x

    bb_nodes = m.getAttr('NodeCount')
    if m.status == GRB.OPTIMAL:
        return m.objVal, X, bb_nodes, m.Runtime, m.MIPGap, True
    else:
        return m.objVal, X, bb_nodes, m.Runtime, m.MIPGap, False

def JS_LP(P, n_machines = None, verbose = False, fixed = []):
    if n_machines == None:
        # You want to solve the problem "as it is"
        n_items, n_machines = P.shape
    else:
        # Your machines are identical and you have m of them, hence, just copy the column vector T m times
        try:
            assert P.shape[1] == 1  # T is a column vector
        except AssertionError:
            raise ValueError("T must be a column vector in the case of identical machines")
        # Copy the column vector T m times
        P = np.tile(P, n_machines)
        n_items = P.shape[0]
    m = gp.Model("exact")
    if not verbose:
        m.setParam('OutputFlag', 0)
        m.setParam('LogToConsole', 0)
    # An integer variable for the makespan
    T = m.addVar(vtype=GRB.CONTINUOUS, name="M")  # Makespan
    # A binary variable x_i_j if variable i is assigned to machine j
    x = m.addVars(n_items, n_machines, vtype=GRB.CONTINUOUS, lb = 0, ub=1, name="x")
    # Constraint: assign each job to exactly one machine
    for i in range(n_items):
        m.addConstr(sum(x[i, j] for j in range(n_machines)) == 1)
    # Constraint: makespan is the maximum completion time
    for j in range(n_machines):
        m.addConstr(sum(x[i, j] * P[i, j] for i in range(n_items)) <= T)
    # Minimize the makespan
    m.setObjective(T, GRB.MINIMIZE)

    # Add the extra constraints
    for (i, j), v in fixed:
        m.addConstr(x[i, j] == v) # v must be either 0 or 1
    m.optimize()
    X = np.zeros((n_items, n_machines))
    for i in range(n_items):
        for j in range(n_machines):
            X[i, j] = x[i, j].x
    return m.objVal, X

def LB(T, P, n_machines = None, verbose = False, fixed = []):
    if n_machines == None:
        raise ValueError("You must specify the number of machines in this case")
    else:
        # Your machines are identical, and you have m of them, hence, just copy the column vector T m times
        try:
            assert P.shape[1] == 1  # P is a column vector
        except AssertionError:
            raise ValueError("T must be a column vector in the case of identical machines")
        # Copy the column vector T m times
        P = np.tile(P, n_machines)
        n_items = P.shape[0]
    m = gp.Model("LB Koppány")
    if not verbose:
        m.setParam('OutputFlag', 0)
        m.setParam('LogToConsole', 0)
    # A binary variable x_i_j if variable i is assigned to machine j
    x = m.addVars(n_items, n_machines, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    # Constraint: assign each job to exactly one machine
    for i in range(n_items):
        m.addConstr(sum(x[i, j] for j in range(n_machines)) == 1)
    # Constraint: makespan is the maximum completion time
    for j in range(n_machines):
        m.addConstr(sum(x[i, j] * P[i, j] for i in range(n_items)) <= T)

    # Add the extra constraints
    for (i, j), v in fixed:
        m.addConstr(x[i, j] == v) # v must be either 0 or 1

    # Optimize the model
    m.setObjective(T, GRB.MINIMIZE) # Constant!
    m.optimize()

    # If the solution is feasible, return True
    if m.status == GRB.OPTIMAL:
        X = np.zeros((n_items, n_machines))
        for i in range(n_items):
            for j in range(n_machines):
                X[i, j] = x[i, j].x
        return True, X
    else:
        return False, None


def list_scheduling_algorithm_identical(P, n_machines = None):
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
    for i in range(n_jobs):
        # Find the machine with the smallest completion time
        j = np.argmin(T)
        X[i, j] = 1
        T[j] += P[i, 0]
    return max(T), X

def JS_LB_BS_identical(P,  n_machines = None, tol = 1e-5, fixed = [], verbose=False):
    '''
    NB: the one you have on the ROSA cluster is correct
    :param P:
    :param n_machines:
    :param tol:
    :param fixed:
    :param verbose:
    :return:
    '''
    all_feas = []
    if n_machines == None:
        raise ValueError("You must specify the number of machines in this case")
    else:
        # Your machines are identical and you have m of them, hence, just copy the column vector T m times
        try:
            assert P.shape[1] == 1  # P is a column vector
        except AssertionError:
            raise ValueError("P must be a column vector in the case of identical machines")
        # Copy the column vector T m times
        P = np.tile(P, n_machines)
        n_jobs = P.shape[0]

    # Define the interval [l, r] where the optimal solution lies
    # Define a clever l
    # Take the sum of the minimum processing time of each job and divide by the number of machines
    l = sum(min(P[i, :]) for i in range(n_jobs)) // n_machines
    # r =  you can simply use the list scheduling algorithm:
    T, X = list_scheduling_algorithm_identical(P, n_machines)
    r = T

    if r == l:
        return r, X, True

    if r - l == 1:
        # Test if l is feasible
        is_feas, X_l = LB(l, P[:, 0].reshape(-1, 1), n_machines, fixed=fixed)
        if is_feas:
            return l, X_l, True
        else:
            return r, X, True

    # Select a new candidate
    T_prime = (l + r) // 2  # Initial solution

    # While the interval is not empty
    while r - l > 1:
        is_feas, X = LB(T_prime, P[:, 0].reshape(-1, 1), n_machines, fixed=fixed)
        if is_feas: # P must be a column vector otherwise you have some complains
            all_feas.append((T_prime, X))
            r = T_prime
        else:
            l = T_prime
        if verbose:
            print("l = ", l, "r = ", r)
        T_prime = (l + r) // 2
    # Return the best feasible solution
    if len(all_feas) == 0:
        return r, X, True
    else:
        best_T, best_X = min(all_feas, key = lambda x: x[0])
        # round the solution at the tolerance level
        return round(best_T, int(-log10(tol))), best_X, False

def list_scheduling_algorithm_unrelated(P):
    '''
     For this it constructs a greedy schedule, in which each job is assigned to the machine on which it has the smallest length.
    :param P:
    :return:
    '''
    n_jobs, n_machines = P.shape
    X = np.zeros((n_jobs, n_machines))
    T = np.zeros(n_machines)
    for j in range(n_jobs):
        # Get the machine with the smallest processing time
        i = np.argmin(P[j, :])
        X[j, i] = 1
        T[i] += P[j, i]
    return max(T), X



def BeB_standard(P, epsilon, n_machines, timelimit = 10*60, verbose = False):
    '''

    :param P:
    :param epsilon:
    :param verbose:
    :return:
    '''
    optimal = False
    depth = 0
    start = time.time()
    T_LB, X_LB, is_optimal = JS_LB_BS_identical(P, n_machines=n_machines, verbose=verbose)
    if not is_optimal:
        X_best = round_LP_solution_matching(X_LB, P, n_machines=n_machines)
        T_max_for_each_machine = []
        for j in range(n_machines):
            T_max_for_each_machine.append(np.dot(P.T, X_best[:, j])[0])
        T_best = max(T_max_for_each_machine)

        # Priority queue for B&B nodes (max-heap based on lower bound, that's why we use -T_LB)
        pq = []

        # Start with the root node
        heapq.heappush(pq, (-T_LB, [], X_LB))

        best_solution = X_best
        best_objective = T_best

        nodes_explored = 0

        start_beb = time.time()
        while pq and time.time() - start_beb < timelimit:
            # Get the node with the lowest bound
            current_lb, fixed_vars, current_solution = heapq.heappop(pq)

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

            else:
                # Branch on the variable that has the largest job
                i = find_largest_fractional(current_solution, P)
                for j in range(n_machines):
                    new_fixed_vars = fixed_vars + [((i, j), 1)]
                    T_new, X_new, is_optimal = JS_LB_BS_identical(P, n_machines=n_machines, fixed=new_fixed_vars, verbose=verbose)
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

    else:
        if verbose:
            print("\t Best solution reached at the root node")
        best_objective = T_LB
        nodes_explored = 1
        optimal = True
    return best_objective, best_lb, best_solution, nodes_explored, depth, time.time() - start, optimal

def BeB_with_profile(P, epsilon, n_machines, timelimit = 60*10, verbose = True):
    '''

    :param P:
    :param epsilon:
    :param verbose:
    :return:
    '''
    # First, get the number of different lengths of P
    start = time.time()
    P_diffs_values = sorted(list(set(P.flatten())))
    different_values = len(set(P.flatten()))
    depth = 0
    T_LB, X_LB, optimal = JS_LB_BS_identical(P, n_machines=n_machines, verbose=False)
    count_discarded = 0
    if not optimal:
        X_best = round_LP_solution_matching(X_LB, P, n_machines=n_machines)
        T_max_for_each_machine = []
        for j in range(n_machines):
            T_max_for_each_machine.append(np.dot(P.T, X_best[:, j])[0])
        T_best = max(T_max_for_each_machine)

        # Priority queue for B&B nodes (max-heap based on lower bound, that's why we use -T_LB)
        pq = []

        # At the root node, the profile is a set of size n_machines having all strings of length different lengths of 0s
        profile = set(['0' * different_values for _ in range(n_machines)])

        # Start with the root node
        heapq.heappush(pq, (-T_LB, [], X_LB, profile))

        best_solution = X_best
        best_objective = T_best

        nodes_explored = 0

        while pq and time.time() - start < timelimit:  # 10 minutes
            # Get the node with the lowest bound
            current_lb, fixed_vars, current_solution, _ = heapq.heappop(pq)

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

            else:
                # Branch on the variable that has the largest job
                i = find_largest_fractional(current_solution, P)
                for j in range(n_machines):
                    new_fixed_vars = fixed_vars + [((i, j), 1)]
                    T_new, X_new, is_optimal = JS_LB_BS_identical(P, n_machines=n_machines, fixed=new_fixed_vars)

                    if T_new < best_objective and not is_optimal:
                        # does it have a different profile? initialize an empty profile
                        profile = ['0' * different_values for _ in range(n_machines)]
                        for (i, j), value in new_fixed_vars:
                            # Get the length of job i
                            length_i = P[i][0]
                            # Get its unique index
                            index_i = P_diffs_values.index(length_i)
                            # Update the profile
                            profile[j] = profile[j][:index_i] + str(int(profile[j][index_i]) + 1) + profile[j][index_i + 1:]
                        # Cast it into a set
                        profile = set(profile)
                        all_profiles = [k[3] for k in pq]
                        if profile not in all_profiles:
                            if verbose:
                                print("Adding a new node with to the queue, current queue: ", len(pq))
                            heapq.heappush(pq, (-T_new, new_fixed_vars, X_new, profile))
                        else:
                            count_discarded += 1
                            if verbose:
                                print("Discarded one node by profile, total nodes discarded: ", count_discarded)
                    else:
                        if verbose:
                            print("Pruned by bound")


            # Exit control: Pick the smallest LB among the active nodes
            if pq:
                best_lb = -pq[0][0]
                X_best_lb = pq[0][2]

            if best_objective / best_lb <= 1 + epsilon:
                if verbose:
                    print("Finished with best objective: ", best_objective, " and best LB: ", best_lb)
                depth = len(fixed_vars)
                optimal = True
                break

    else:
        if verbose:
            print("\t Best solution reached at the root node")
        best_objective = T_LB
        best_solution = X_LB
        X_best_lb = X_LB
        nodes_explored = 1
        optimal = True
    return best_objective, X_best_lb, best_solution, nodes_explored, depth, time.time() - start, optimal



