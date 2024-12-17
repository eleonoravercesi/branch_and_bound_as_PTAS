import numpy as np
from fractions import Fraction
import networkx as nx

def is_integer_job(X, tol = 1e-5):
    m = X.shape[0]
    for j in range(m):
        if abs(X[j] - round(X[j])) > tol:
            return False
    return True

def is_integer_sol(x, tol = 1e-5):
    n, m = x.shape
    for i in range(n):
        if not is_integer_job(x[i, :], tol):
            return False
    return True

def find_largest_fractional(X, P):
    n, m = X.shape
    not_integer = []
    for i in range(n):
        if not is_integer_job(X[i, :]):
            not_integer.append(i)
    return max(not_integer, key = lambda i: Fraction(P[i, 0]))

def round_LP_solution_matching(X, P, n_machines = None):
    '''
    Round LP solution using minimum weight matching
    :param X:
    :return:
    '''
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
        n_jobs = P.shape[0]
    fractional_jobs = []
    for i in range(n_jobs):
        if not is_integer_job(X[i, :]):
            fractional_jobs.append(i)

    # Create a bipartite graph between the jobs and the machines
    G = nx.Graph()
    for j in range(n_machines):
        G.add_node("m_" + str(j))
    for i in fractional_jobs:
        G.add_node("j_" + str(i))
        for j in range(n_machines):
            G.add_edge("m_" + str(j), "j_" + str(i), weight = P[i, j])

    # Solve the minimum weight matching problem
    matching = nx.min_weight_matching(G)
    X_rounded = np.zeros((n_jobs, n_machines))
    for (A, B) in matching:
        if A.startswith("j_"):
            i = int(A[2:])
            j = int(B[2:])
        else:
            i = int(B[2:])
            j = int(A[2:])
        X_rounded[i, j] = 1
    for i in range(n_jobs):
        if i not in fractional_jobs:
            X_rounded[i, :] = X[i, :]
    return X_rounded