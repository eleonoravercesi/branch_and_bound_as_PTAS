import numpy as np
from ortools.linear_solver import pywraplp

def identical_machines_job_scheduling(p, n_machines, timelimit = 10*60, tol = 1e-6, verbose = False):
    """
    Solve the identical machines job scheduling problem using Google OR-Tools.

    Parameters:
    - jobs: List of integers representing the processing times of jobs.
    - machines: Integer representing the number of identical machines.
    - timelimit: Time limit in seconds.

    Returns:
    - Optimal makespan (Cmax).
    - Job assignments to machines.
    """
    # Create the solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    # Set a timelimit
    solver.set_time_limit(timelimit * 1000)  # Convert seconds to milliseconds
    if not solver:
        print("Solver not available.")
        return None, None

    if not verbose:
        solver.SuppressOutput()

    # Problem data
    n = len(p)  # Number of jobs
    m = n_machines  # Number of machines

    # Decision variables
    # x[j, i] = 1 if job j is assigned to machine i, 0 otherwise
    x = {}
    for j in range(n):  # Outer loop over jobs
        for i in range(m):  # Inner loop over machines
            x[j, i] = solver.BoolVar(f"x_{j}_{i}")

    # Cmax: Makespan (maximum load on any machine)
    Cmax = solver.NumVar(0, solver.infinity(), "Cmax")

    # Objective: Minimize Cmax
    solver.Minimize(Cmax)

    # Constraints:
    # 1. Each job is assigned to exactly one machine
    for j in range(n):
        solver.Add(sum(x[j, i] for i in range(m)) == 1)

    # 2. The total load on each machine cannot exceed Cmax
    for i in range(m):
        solver.Add(sum(p[j] * x[j, i] for j in range(n)) <= Cmax)

    # Solve the problem
    status = solver.Solve()

    # Check solution status
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        makespan = Cmax.solution_value()
        X = {}
        for j in range(n):
            for i in range(m):
                if x[j, i].solution_value() > tol:
                    X[(j, i)] = x[j, i].solution_value()
        # Return also the timelimit
        return makespan, X,  status
    else:
        return None, None, None


