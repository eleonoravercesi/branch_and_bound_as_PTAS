import numpy as np
import gurobipy as gp
from gurobipy import GRB
from algorithms import JS_LP
from itertools import combinations
from math import ceil
import subprocess
from itertools import combinations

TOL = 1e-6


def get_unfixed(X):
    n, m = X.shape
    unfixed = []
    for i in range(n):
        for j in range(m):
            if abs(round(X[i, j]) - X[i, j]) > TOL:
                unfixed.append(i)
                break
    return unfixed


def from_A_b_to_polymake_file(A, b, index_of_last_equality_constraint, out_filename):
    F = open(out_filename, "w+")
    F.write('use application "polytope";\nmy $p = new Polytope(EQUATIONS=>[')
    n_constraints = A.shape[0]
    n_vars = A.shape[1]
    A_eq = A[:index_of_last_equality_constraint, :]
    b_eq = b[:index_of_last_equality_constraint, :]
    A_ineq = A[index_of_last_equality_constraint:, :]
    b_ineq = b[index_of_last_equality_constraint:, :]

    for k in range(A_eq.shape[0]):
        F.write("[{}, ".format(str(-b_eq[k][0])))
        contraint = ", ".join([str(A_eq[k, j]) for j in range(n_vars)])
        F.write(contraint)
        if k == A_eq.shape[0] - 1:
            F.write("]")
        else:
            F.write("], ")

    F.write('], INEQUALITIES=>[')
    # Write constraints (inequalities)

    for i in range(A_ineq.shape[0]):
        F.write("[{}, ".format(str(-b_ineq[i][0])))
        contraint = ", ".join([str(A_ineq[i, j]) for j in range(n_vars)])
        F.write(contraint)
        if i == n_constraints - 1:
            F.write("]")
        else:
            F.write("], ")
    F.write("]);\n")
    F.write("print($p->VERTICES);")
    F.close()


def JS_LP_A_b(P, T, n_machines=None, verbose=False, fixed=[]):
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
    # A binary variable x_i_j if variable i is assigned to machine j
    x = m.addVars(n_items, n_machines, vtype=GRB.CONTINUOUS, name="x")
    # Constraint: assign each job to exactly one machine
    for i in range(n_items):
        m.addConstr(sum(x[i, j] for j in range(n_machines)) == 1)

    # Add the extra constraints
    for (i, j), v in fixed:
        m.addConstr(x[i, j] == v)  # v must be either 0 or 1

    # Constraint: makespan is the maximum completion time
    for j in range(n_machines):
        m.addConstr(sum(- [i, j] * P[i, j] for i in range(n_items)) >= -T)

    for i in range(n_items):
        for j in range(n_machines):
            m.addConstr(x[i, j] >= 0)

    return m.getA(), m.getAttr('RHS')


number_of_seeds = 1
n_jobs = 3
n_machines = 2

pairs_to_idx = dict(zip([(i, j) for i in range(n_jobs) for j in range(n_machines)], range(n_jobs * n_machines)))
idx_to_pairs = dict(zip(range(n_jobs * n_machines), [(i, j) for i in range(n_jobs) for j in range(n_machines)]))

for seed in range(1):
    np.random.seed(seed)
    P = np.random.randint(1, 100, (n_jobs, n_machines))


    T, X_s = JS_LP(P)

    T_int = ceil(T)
    A, b = JS_LP_A_b(P, T_int)

    # From scipy sparse to numpy
    A = A.toarray()
    b = np.asarray(b).reshape(-1, 1)

    index_of_last_equality_constraint = P.shape[0]

    from_A_b_to_polymake_file(A, b, index_of_last_equality_constraint, out_filename="./output/vertices.pl")

    # Run polymake
    subprocess.run("polymake --script ./output/vertices.pl > ./output/vertices.txt", shell=True)

    # # Parse the output
    # with open("./output/vertices.txt", "r") as f:
    #     lines = f.readlines()

    #vertices = []

    # for line in lines:
    #     v = []
    #     line_vec = line.strip().split(" ")
    #     for v_i in line_vec[1:]:
    #         if "/" in v_i:
    #             v_i = v_i.split("/")
    #             v.append(int(v_i[0]) / int(v_i[1]))
    #         else:
    #             v.append(int(v_i))
    #     vertices.append(v)
    #
    # # Are these vertices a good counter example?
    # all_pairs = combinations(range(n_jobs), 2)
    # for pair in all_pairs:
    #     v_1, v_2 = vertices[pair[0]], vertices[pair[1]]
    #     # Do they have a common 1?
    #     for i1 in range(n_jobs):
    #         for j1 in range(n_machines):
    #             if v_1[pairs_to_idx[(i1, j1)]] == 1 and v_2[pairs_to_idx[(i1, j1)]] == 1:
    #                 # They have the same job assigned on two different machines
    #                 # Do they have another fractional jon i common
    #                 for i2 in range(n_jobs):
    #                     if i2 != i1:
    #                         for j2 in range(n_machines):
    #                             # Do they have a common non zero?
    #                             if v_1[pairs_to_idx[(i2, j2)]] > TOL and v_2[pairs_to_idx[(i2, j2)]] != TOL:
    #                                 print("Found a counter example")
    #                                 print(P)
    #                                 print(v_1)
    #                                 print(v_2)
    #                                 exit(1)