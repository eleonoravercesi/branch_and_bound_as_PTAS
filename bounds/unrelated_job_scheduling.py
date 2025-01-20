from pyscipopt import Model

import main_multi_knapsack
from DEBUG import n_items


def binary_search(completions_times, verbose=False):
    pass


def linear_relaxation(completions_times, initial_makespan, fixed, verbose=False):
    """
    Parameters
    ----------
    completions_times : list
        List of lists completion times for each job on each machine. p[j][i] is the completion time of job j on machine i.
    initial_makespan : list
        List with all the initial makespan, one for each machine
    fixed : list
        List of tuples with the fixed items, (j, i) --> item j is fixed on machine i
    """
    model = Model("Unrelated Job Scheduling")

    if not verbose:
        model.hideOutput()

    n_jobs = len(completions_times)
    n_machines = len(completions_times[0])

    unfixed_jobs = [j for j in range(n_jobs) if j not in [j for j, i in fixed]]


    # Decision variables
    x = {}
    for j in unfixed_jobs:
        for i in range(n_machines):
            x[j, i] = model.addVar(vtype="C", name=f"x({i},{j})")

    # Makespan
    C_max = model.addVar(vtype="C", name="C_max")

    # Objective function
    model.setObjective(C_max, "minimize")

    # Constraint 1. You have to allocate each job
    for j in unfixed_jobs:
        model.addCons(sum(x[j, i] for i in range(n_machines)) == 1)

    # Constraint 2. The completion time on each machine must be at most C_max
    for i in range(n_machines):
        model.addCons(sum(x[j, i] * completions_times[j][i] for j in unfixed_jobs) <= C_max - initial_makespan[i])


    # Optimize the model
    model.optimize()

    # Extract the solution
    if model.getStatus() == "optimal":
        solution = {(j, i): model.getVal(x[(j, i)]) for j in unfixed_jobs for i in
                    range(n_machines) if model.getVal(x[(j, i)]) > 0}
        return solution, model.getVal(), True
    else:
        return None, None, False # Let's keep it, but it's always feasible