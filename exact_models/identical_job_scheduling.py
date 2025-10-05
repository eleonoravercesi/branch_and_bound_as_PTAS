from ortools.linear_solver import pywraplp
from pyscipopt import Model


def solve_identical_job_scheduling(n_jobs: int, n_machines: int, processing_times: list[int], verbose=False):
    """
        Solves the Identical Job Scheduling problem using OR-Tools with SCIP backend.

        Parameters:
            n_jobs (int): Number of jobs to be scheduled.
            n_machines (int): Number of identical machines available.
            processing_times (list): List of processing times for each job (uniform across machines).

        Returns:
            dict: Solution with assigned jobs and total profit.
        """
    assert n_jobs == len(processing_times), "Number of jobs must match the length of processing_times"
    assert n_machines > 0, "There must be at least one machine"

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
        solver.Add(solver.Sum(processing_times[j] * x[j, i] for j in range(n_jobs)) <= C_max)

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
