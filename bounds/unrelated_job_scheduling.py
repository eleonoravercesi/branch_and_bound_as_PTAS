from pyscipopt import Model
import math
from utils import is_integer_val


def binary_search(processing_times, initial_makespan, fixed, verbose=False):
    """
        Parameters
        ----------
        processing_times : list
            List of lists of processing times for each job on each machine.
            p[j][i] is the processing time of job j on machine i.
        initial_makespan : list
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

    # Now we imitate the binary search.
    # The value is simply the upper integer part of the linear relaxation
    _, bin_search_lb, _ = linear_relaxation(processing_times, initial_makespan, fixed, False)
    bin_search_lb = math.ceil(bin_search_lb)

    "Now we find a solution with this makespan"
    # Decision variables
    x = {}
    for j in unfixed_jobs:
        for i in range(n_machines):
            if processing_times[j][i] <= bin_search_lb:
                x[j, i] = model.addVar(vtype="C", name=f"x({i},{j})", lb=0.0)
            else:
                x[j, i] = model.addVar(vtype="C", name=f"x({i},{j})", lb=0.0, ub=0.0)

    # Constraint 1. You have to allocate each job
    for j in unfixed_jobs:
        model.addCons(sum(x[j, i] for i in range(n_machines)) == 1)

    # Constraint 2. The completion time on each machine must be at most C_max
    for i in range(n_machines):
        model.addCons(sum(x[j, i] * processing_times[j][i] for j in unfixed_jobs) <= bin_search_lb-initial_makespan[i])

    # Set the solver method to simplex (for having a feas. sol. that's a vertex)
    model.setParam("lp/resolvealgorithm", 'p')

    # Optimize the model
    model.optimize()

    # Extract the solution
    if model.getStatus() == "optimal":
        solution = {(j, i): model.getVal(x[(j, i)]) for j in unfixed_jobs for i in
                    range(n_machines) if model.getVal(x[(j, i)]) > 0}
        assert len([j for (j, i) in solution.keys() if not is_integer_val(solution[(j, i)])]) <= n_machines, "Too many fractional jobs"
        return solution, bin_search_lb, True
    else:
        return None, None, False  # Let's keep it, but it's always feasible


def linear_relaxation(processing_times, initial_makespan, fixed, verbose=False):
    """
    Parameters
    ----------
    processing_times : list
        List of lists completion times for each job on each machine.
        p[j][i] is the completion time of job j on machine i.
    initial_makespan : list
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
        model.addCons(sum(x[j, i] * processing_times[j][i] for j in unfixed_jobs) <= C_max - initial_makespan[i])

    # Set the solver method to simplex (for having a feas. sol. that's a vertex)
    model.setParam("lp/resolvealgorithm", 'p')

    # Optimize the model
    model.optimize()

    # Extract the solution
    if model.getStatus() == "optimal":
        solution = {(j, i): model.getVal(x[(j, i)]) for j in unfixed_jobs for i in
                    range(n_machines) if model.getVal(x[(j, i)]) > 0}
        makespan = model.getObjVal()
        assert len([j for (j, i) in solution.keys() if not is_integer_val(solution[(j, i)])]) <= n_machines, "Too many fractional jobs"
        return solution, makespan, True
    else:
        return None, None, False  # Let's keep it, but it's always feasible
