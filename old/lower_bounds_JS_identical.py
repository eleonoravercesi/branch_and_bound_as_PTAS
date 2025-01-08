from pyscipopt import Model, quicksum
from heuristics.job_scheduling import list_scheduling_algorithm_identical
from math import log10


def linear_relaxation(p, n_machines, fixed_assignments=None, check_feasibility = 0, tol=1e-6, verbose = False):
    """
    Compute the linear relaxation of the identical machines job scheduling problem.

    Parameters:
    - jobs: List of integers representing the processing times of the jobs.
    - machines: Integer representing the number of identical machines.
    - fixed_assignments: List of tuples (job, machine) with fixed job assignments.
    - tol: Tolerance for comparing floating point numbers.
    - verbose: Print information about the optimization process.

    Returns:
    - Objective value of the linear relaxation (makespan).
    - Dictionary with job assignments (fractional values).
    """
    model = Model("Linear_Relaxation_Identical_Machines")
    # Set the verbosity of the solver
    if not verbose:
        model.hideOutput()

    # Parameters
    n = len(p)  # Number of jobs
    m = n_machines  # Number of identical machines

    # Variables
    # x[j][i] = Fractional assignment of job j to machine i
    x = {}
    for j in range(n):
        for i in range(m):
            x[(j, i)] = model.addVar(vtype="CONTINUOUS", name=f"x_{j}_{i}", lb=0, ub=1)


    # Constraints:
    # 1. Each job is assigned to exactly one machine (relaxed to fractional assignment)
    for j in range(n):
        model.addCons(quicksum(x[(j, i)] for i in range(m)) == 1, name=f"job_assignment_{j}")

    # 2. The load on each machine cannot exceed the makespan
    if check_feasibility == 0:
        # Makespan is a variable!!
        T = model.addVar(vtype="CONTINUOUS", name="Cmax")
        for i in range(m):
            model.addCons(quicksum(p[j] * x[(j, i)] for j in range(n)) <= T, name=f"machine_load_{i}")
            # Objective: Minimize Cmax
        model.setObjective(T, sense="minimize")
    else:
        T = check_feasibility
        for i in range(m):
            model.addCons(quicksum(p[j] * x[(j, i)] for j in range(n)) <= T, name=f"machine_load_{i}")

    # 3. Fixed job assignments
    if fixed_assignments is not None:
        for j, i in fixed_assignments:
            model.addCons(x[(j, i)] == 1, name=f"fixed_assignment_{j}_{i}")

    # Solve the LP relaxation
    model.optimize()

    # Check solution status
    X = {}
    if model.getStatus() == "optimal":
        for j in range(n):
            for i in range(m):
                # Get the solution of x[j][i]
                if model.getVal(x[(j, i)]) > tol:
                    X[(j, i)] = model.getVal(x[(j, i)])
        return model.getObjVal(), X, True

    else:
        print(f"Solver status: {model.getStatus()}")
        return None, None, False # Should never be infeasible, but ok for now





def binary_search(P, n_machines, fixed_assignments=None, tol=1e-6, verbose = False):
    all_feas = []

    # Define the initial search interval
    l = sum(P) // n_machines  # Clever lower bound
    T, X = list_scheduling_algorithm_identical(P, n_machines)
    r = T  # Upper bound from list scheduling

    # Check if lower and upper bounds are already optimal
    if r == l:
        return r, X, True

    # Edge case: when r - l == 1, explicitly check both bounds
    if r - l == 1:
        _,  X_l, is_feas = linear_relaxation(P, n_machines, fixed_assignments, check_feasibility=l, verbose=False)
        if is_feas:
            return l, X_l, True
        else:
            return r, X, True

    # Perform binary search
    while r - l > 1:
        T_prime = (l + r) // 2  # Midpoint candidate
        _,  X_l, is_feas = linear_relaxation(P, n_machines, fixed_assignments, check_feasibility=T_prime)

        if is_feas:
            # Store feasible solutions
            all_feas.append((T_prime, X))
            r = T_prime  # Update upper bound
        else:
            l = T_prime  # Update lower bound

        if verbose:
            print(f"l = {l}, r = {r}, T_prime = {T_prime}")

    # Final feasibility check for the lower bound l
    _,  X_l, is_feas = linear_relaxation(P, n_machines, fixed_assignments, check_feasibility=l)
    if is_feas:
        all_feas.append((l, X_l))

    # Final feasibility check for the upper bound r
    _,  X_r, is_feas = linear_relaxation(P, n_machines, fixed_assignments, check_feasibility=r)
    if is_feas:
        all_feas.append((r, X_r))

    # Return the best feasible solution
    if len(all_feas) == 0:
        return r, X, False  # No feasible solutions found
    else:
        best_T, best_X = min(all_feas, key=lambda x: x[0])
        return round(best_T, int(-log10(tol))), best_X, True

