from ortools.linear_solver import pywraplp
from pyscipopt import Model, SCIP_PARAMSETTING


def solve_unrelated_job_scheduling(processing_times, verbose=False):
    """
        Solves the Unrelated Job Scheduling problem using OR-Tools with SCIP backend.

        Parameters:
            processing_times (list): List of processing times for each job on each machine.

        Returns:
            dict: Solution with assigned jobs and total profit.
        """
    n_jobs = len(processing_times)  # Number of jobs, indexed with j
    n_machines = len(processing_times[0])  # Number of machines, indexed with i

    # Initialize OR-Tools Solver with SCIP backend
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception('SCIP solver not available.')

    # Define decision variables
    x = {}
    for j in range(n_jobs):
        for i in range(n_machines):
            x[j, i] = solver.BoolVar(f'x_{i}_{j}')

    # Makespan
    C_max = solver.NumVar(0, solver.infinity(), 'C_max')

    # Objective Function: Maximize total profit
    solver.Minimize(C_max)

    # Constraint 1. You have to allocate each job
    for j in range(n_jobs):
        solver.Add(solver.Sum(x[j, i] for i in range(n_machines)) == 1)

    # Constraint 2: The processing time on each machine must be at most C_max
    for i in range(n_machines):
        solver.Add(solver.Sum(processing_times[j][i] * x[j, i] for j in range(n_jobs)) <= C_max)

    # Solve the problem
    status = solver.Solve()

    # Get the final runtime
    runtime = solver.wall_time()

    if status == pywraplp.Solver.OPTIMAL:
        if verbose:
            print('Optimal solution found!')
        makespan = solver.Objective().Value()
        assignment = {}
        for i in range(n_jobs):
            for j in range(n_machines):
                if x[i, j].solution_value() > 0.5:
                    assignment[(i, j)] = 1
        return makespan, assignment, status, runtime
    else:
        print('No optimal solution found.')
        return None, None, status, runtime


def SCIP(processing_times, verbose=False):
    """
    Compute an optimal solution using SCIP B&B with no cutting plane and presolve
    """

    # Initialize the model
    model = Model("Knapsack")

    if not verbose:
        model.hideOutput()

    # Configure simple Branch-and-Bound
    model.setIntParam("presolving/maxrounds", 0)  # Disable presolve
    model.setIntParam("separating/maxrounds", 0)  # Disable cutting planes
    model.setHeuristics(SCIP_PARAMSETTING.OFF)  # Disable heuristics
    model.setIntParam("presolving/maxrestarts", 0)  # Disable restarts
    model.setIntParam("propagating/maxrounds", 0)  # Disable propagation

    n_machines = len(processing_times[0])
    n_jobs = len(processing_times)

    # Decision variables
    x = {}
    for j in range(n_jobs):
        for i in range(n_machines):
            x[j, i] = model.addVar(vtype="B", name=f"x({i},{j})")

    # Makespan
    C_max = model.addVar(vtype="C", name="C_max")

    # Objective function
    model.setObjective(C_max, "minimize")

    # Constraint 1. You have to allocate each job
    for j in range(n_jobs):
        model.addCons(sum(x[j, i] for i in range(n_machines)) == 1)

    # Constraint 2. The processing time on each machine must be at most C_max
    for i in range(n_machines):
        model.addCons(sum(x[j, i] * processing_times[j][i] for j in range(n_jobs)) <= C_max)

    # Optimize the model
    model.optimize()

    # Extract the solution
    if model.getStatus() == "optimal":
        solution = {(j, i): model.getVal(x[(j, i)]) for j in range(n_jobs) for i in
                    range(n_machines) if model.getVal(x[(j, i)]) > 0}
        return solution, model.getVal(), True
    else:
        return None, None, False  # Let's keep it, but it's always feasible
