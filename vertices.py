import numpy as np
import gurobipy as gp
from gurobipy import GRB
from algorithms import JS_LP
import fractions

def JS_LP_A_b(P, T, n_machines = None, verbose = False, fixed = []):
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
        m.addConstr(sum(x[i, j] * P[i, j] for i in range(n_items)) <= T)

    for i in range(n_items):
        for j in range(n_machines):
            m.addConstr(x[i, j] >= 0)

    return m.getA(), m.getAttr('RHS')

P = np.array([[48, 69],
[26, 68],
[84, 24]])

obj, X = JS_LP(P)
print(obj)
from math import ceil
obj = ceil(obj)
A, b = JS_LP_A_b(P, obj)

# From scipy sparse to numpy
A = A.toarray()
b = np.asarray(b).reshape(-1, 1)

index_of_last_equality_constraint = P.shape[0]

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



from_A_b_to_polymake_file(A,b, index_of_last_equality_constraint, out_filename="./output/vertices.pl")