from pyscipopt import Model, SCIP_PARAMSETTING
from utils import is_integer_val


def solve_greedy(n_jobs: int, n_machines: int, processing_times: list[int], overhead, fixed_jobs, verbose=False):
    """
        Parameters
        ----------
        n_jobs : int
            Number of jobs to be scheduled.
        n_machines : int
            Number of identical machines available.
        processing_times : list
            List of processing times for each job.
        overhead : list
            List with all completion times, one for each machine
        fixed_jobs : list
            List of tuples with the fixed items, (j, i) --> item j is fixed on machine i

        Returns
        -------
        solution_fractional : dict
            Dictionary with the solution, (j, i): f means job j is assigned to machine i with fraction f
        C_max : float
            The makespan of the solution
        is_feasible : bool
            True if the solution is feasible, False otherwise
    """

    # Initialize the completion times for each machine
    completion_times = overhead.copy()

    # Find the machine (i_max) that determines the makespan (C_max) and store both its index and completion time
    i_max, C_max = max(enumerate(completion_times), key=lambda t: t[1])

    # List of unfixed jobs
    unfixed_jobs = [j for j in range(n_jobs) if j not in [j for j, i in fixed_jobs]]
    
    # Compute the amount of processing time available on each machine.
    # Remarks:
    # - processing time available is to be understood as the amount of processing time that can be
    #   allocated to a machine without exceeding the current makespan C_max, hence without changing it
    # - the machine that determines the makespan has 0 processing time available
    available_times = [C_max - completion_times[i] for i in range(n_machines)]

    # Compute the total available time across all machines
    total_available_time = sum(available_times)

    # Compute the total processing time of unfixed jobs
    total_processing_time = sum(processing_times[j] for j in unfixed_jobs)

    C_additional = 0    # Additional time to the makespan if needed
    if total_processing_time > total_available_time:
        # In this case the new makespan will be C_max + C_additional
        C_additional = (total_processing_time - total_available_time) / n_machines
        available_times = [available_times[i] + C_additional for i in range(n_machines)]

    # Assign unfixed jobs to machines using a greedy approach
    solution_fractional = {}
    for j in unfixed_jobs:
        C_j = processing_times[j]
        C_j_remaining = C_j
        
        if C_j == 0:
            # If the job has zero processing time, assign it completely to the machine with the smallest index
            # This edge case shouldn't happen in practice, but we handle it for the sake of completeness
            solution_fractional[j, 0] = 1
            continue
        
        for i in range(n_machines):
            if available_times[i] > 0:
                if C_j_remaining <= available_times[i]:
                    # Assign job j to machine i
                    solution_fractional[j, i] = C_j_remaining / C_j
                    available_times[i] -= C_j_remaining
                    break
                else:
                    # Assign a fraction of job j to machine i
                    solution_fractional[j, i] = available_times[i] / C_j
                    C_j_remaining -= available_times[i]
                    available_times[i] = 0
    
    # Verify post-conditions
    assert abs(sum(solution_fractional[(j, i)] for (j, i) in solution_fractional.keys())-len(unfixed_jobs)) < 1e-9, "Some unfixed jobs have not been completely assigned, or are over-assigned."

    return solution_fractional, C_max + C_additional, True


def binary_search(n_jobs: int, n_machines: int, processing_times: list[int], overhead, fixed, verbose=False):
    """
        Parameters
        ----------
        n_jobs : int
            Number of jobs to be scheduled.
        n_machines : int
            Number of identical machines available.
        processing_times : list
            List of processing times for each job.
        overhead : list
            List with all the initial makespan, one for each machine
        fixed : list
            List of tuples with the fixed items, (j, i) --> item j is fixed on machine i
    """

    unfixed_jobs = [j for j in range(n_jobs) if j not in [j for j, i in fixed]]

    """
    Now we do the binary search. The possible makespans are in the interval (left,right].
    In other words, "left" is never feasible, "right" is always feasible.
    We can initialize "left" as the largest minimal processing time - 1 (it can never be a makespan),
    and "right" as the assignment where each job is allocated to the machine with smallest overhead (it's always feasible)
    """
    left = max(min(processing_times[j] + overhead[i] for i in range(n_machines)) for j in unfixed_jobs) - 1

    shortest_machine = min([i for i in range(n_machines)], key=lambda t: overhead[t])
    temp_completion_times = overhead.copy()
    temp_completion_times[shortest_machine] += sum(processing_times[j] for j in unfixed_jobs)
    right = max(temp_completion_times)
    solution = {(j, shortest_machine): 1 for j in unfixed_jobs}

    while right - left > 1:
        middle = (left + right) // 2
        model = Model("Unrelated Job Scheduling")
        model.setParam("lp/resolvealgorithm", 'p')
        model.setParam('lp/initalgorithm', 'p')  # Use primal simplex for the initial LP
        model.setParam('lp/resolvealgorithm', 'p')  # Use primal simplex for re-solves

        # Configure simple Branch-and-Bound
        model.setIntParam("presolving/maxrounds", 0)  # Disable presolve
        model.setIntParam("separating/maxrounds", 0)  # Disable cutting planes
        model.setHeuristics(SCIP_PARAMSETTING.OFF)  # Disable heuristics
        model.setIntParam("presolving/maxrestarts", 0)  # Disable restarts
        model.setIntParam("propagating/maxrounds", 0)  # Disable propagation

        if not verbose:
            model.hideOutput()
        else:
            model.setParam('display/verblevel', 5)

        # Decision variables
        x = {}
        for j in unfixed_jobs:
            for i in range(n_machines):
                if processing_times[j] <= middle:
                    x[j, i] = model.addVar(vtype="C", name=f"x({i},{j})", lb=0.0)
                else:
                    x[j, i] = model.addVar(vtype="C", name=f"x({i},{j})", lb=0.0, ub=0.0)

        # Constraint 1. You have to allocate each job
        for j in unfixed_jobs:
            model.addCons(sum(x[j, i] for i in range(n_machines)) == 1)

        # Constraint 2. The completion time on each machine must be at most C_max
        for i in range(n_machines):
            model.addCons(sum(x[j, i] * processing_times[j] for j in unfixed_jobs) <= middle - overhead[i])

        model.setObjective(x[unfixed_jobs[0], 0], "minimize")

        # Optimize the model
        model.optimize()

        # Extract the solution
        if model.getStatus() == "optimal":
            solution = {(j, i): model.getVal(x[(j, i)]) for j in unfixed_jobs for i in
                        range(n_machines) if model.getVal(x[(j, i)]) > 0}
            assert len(list(set([j for (j, i) in solution.keys() if not is_integer_val(solution[(j, i)])]))) <= n_machines, "Too many fractional jobs"
            right = middle
        else:
            left = middle

    # The optimal makespan is always "right" at the last iteration
    return solution, right, True


def linear_relaxation(processing_times, overhead, fixed, verbose=False):
    """
    Parameters
    ----------
    processing_times : list
        List of lists completion times for each job on each machine.
        p[j][i] is the completion time of job j on machine i.
    overhead : list
        List with all the initial makespan, one for each machine
    fixed : list
        List of tuples with the fixed items, (j, i) --> item j is fixed on machine i
    """
    model = Model("Unrelated Job Scheduling")

    if not verbose:
        model.hideOutput()

    n_jobs = len(processing_times)
    n_machines = len(processing_times[0])

    unfixed_jobs = [j for j in range(n_jobs) if j not in [j for j, i in fixed]]

    # Decision variables
    x = {}
    for j in unfixed_jobs:
        for i in range(n_machines):
            x[j, i] = model.addVar(vtype="C", name=f"x({i},{j})", lb=0.0)

    # Makespan
    C_max = model.addVar(vtype="C", name="C_max", lb=0.0)

    # Objective function
    model.setObjective(C_max, "minimize")

    # Constraint 1. You have to allocate each job
    for j in unfixed_jobs:
        model.addCons(sum(x[j, i] for i in range(n_machines)) == 1)

    # Constraint 2. The completion time on each machine must be at most C_max
    for i in range(n_machines):
        model.addCons(sum(x[j, i] * processing_times[j][i] for j in unfixed_jobs) <= C_max - overhead[i])

    # Set the solver method to simplex (for having a feas. sol. that's a vertex)
    model.setParam("lp/resolvealgorithm", 'p')

    # Optimize the model
    model.optimize()

    # Extract the solution
    if model.getStatus() == "optimal":
        solution = {(j, i): model.getVal(x[(j, i)]) for j in unfixed_jobs for i in
                    range(n_machines) if model.getVal(x[(j, i)]) > 0}
        makespan = model.getObjVal()
        assert len(set([j for (j, i) in solution.keys() if not is_integer_val(solution[(j, i)])])) <= n_machines, "Too many fractional jobs"
        return solution, makespan, True
    else:
        return None, None, False  # Let's keep it, but it's always feasible
