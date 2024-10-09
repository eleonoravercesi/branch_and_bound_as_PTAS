import gurobipy as gp
from gurobipy import GRB
import numpy as np
from math import log10

def JS_ILP(P, n_machines = None, verbose = False, presolve = True, cuts = True, gap = None, timelimit = None):
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
        return m.objVal, X, bb_nodes, True
    else:
        return m.objVal, X, bb_nodes, False

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
        # Your machines are identical and you have m of them, hence, just copy the column vector T m times
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

    :param P:
    :param n_machines:
    :param tol:
    :param fixed:
    :param verbose:
    :param is_root: if True, you can return when r - l < 1. More specifically, you can return r.
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
    l = sum(min(P[i, :]) for i in range(n_jobs)) // n_machines # Integer division bc we want them integer
    # r =  you can simply use the list scheduling algorithm:
    T, X = list_scheduling_algorithm_identical(P, n_machines)
    r = T

    if r == l:
        return r, X, True


    # Select a new candidate
    # If l - r equals 1, this is exactly l
    T_prime = (l + r) // 2  # Initial solution
    first_iteration = True

    # While the interval is not empty
    while r - l >= 1:
        is_feas, X = LB(T_prime, P[:, 0].reshape(-1, 1), n_machines, fixed=fixed)
        if first_iteration and is_feas:
            return T_prime, X, True # Return the optimal solution that is actually r with is fesible assignment
        first_iteration = False # After the first iteration, set this variable to false
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
        return None, None, None
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


def JS_LB_BS_fast_unrelated(P, tol = 1e-5, fixed = [], verbose=False, is_root=False):
    '''

    :param P:
    :param n_machines:
    :param tol:
    :param fixed:
    :param verbose:
    :param is_root: if True, you can return when r - l < 1. More specifically, you can return r.
    :return:
    '''
    n_jobs, n_machines = P.shape

    # Define the interval [l, r] where the optimal solution lies
    # Define a clever l
    # r =  you can simply use the list scheduling algorithm:
    T, X = list_scheduling_algorithm_unrelated(P)
    r = T
    # Left is the former value divided by the number of machines
    l = T / n_machines

    if 0 <= r - l <= 1:
        # TODO fix here
        if is_root: # you can return r
            return r, X, True # Last boolen value is true meaning that it's optimal! This is important
        if int(l) == l: # l is an integer
            return int(l), # TODO
        return T, X

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
        return None, None
    else:
        best_T, best_X = min(all_feas, key = lambda x: x[0])
        # round the solution at the tolerance level
        return round(best_T, int(-log10(tol))), best_X