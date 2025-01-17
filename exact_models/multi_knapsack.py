from ortools.linear_solver import pywraplp
from pyscipopt import Model, SCIP_PARAMSETTING


def solve_multi_knapsack(profits, weights, capacities, verbose=False):
    """
    Solves the Multi-Knapsack Problem using OR-Tools with SCIP backend.

    Parameters:
        profits (list): List of item profits.
        weights (list): List of item weights.
        capacities (list): List of knapsack capacities.

    Returns:
        dict: Solution with assigned items and total profit.
    """
    num_items = len(profits)  # Number of items, indexed with j
    num_knapsacks = len(capacities)  # Number of knapsacks, indexed with i

    # Initialize OR-Tools Solver with SCIP backend
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception('SCIP solver not available.')

    # Define decision variables
    x = {}
    for j in range(num_items):
        for i in range(num_knapsacks):
            x[j, i] = solver.BoolVar(f'x_{i}_{j}')

    # Objective Function: Maximize total profit
    solver.Maximize(solver.Sum(profits[j] * x[j, i] for j in range(num_items) for i in range(num_knapsacks)))

    # Constraint 1: Each item can be assigned to at most one knapsack
    for j in range(num_items):
        solver.Add(solver.Sum(x[j, i] for i in range(num_knapsacks)) <= 1)

    # Constraint 2: Knapsack capacity constraints
    for i in range(num_knapsacks):
        solver.Add(solver.Sum(weights[j] * x[j, i] for j in range(num_items)) <= capacities[i])

    # Solve the problem
    status = solver.Solve()

    # Get the final runtime
    runtime = solver.wall_time()

    if status == pywraplp.Solver.OPTIMAL:
        if verbose:
            print('Optimal solution found!')
        total_profit = solver.Objective().Value()
        assignment = {}
        for i in range(num_items):
            for j in range(num_knapsacks):
                if x[i, j].solution_value() > 0.5:
                    assignment[(i, j)] = 1
        return total_profit, assignment, status, runtime
    else:
        print('No optimal solution found.')
        return None, None, status, runtime

def SCIP(profits, weights, capacities):
    """
    Compute an optimal solution using SCIP B&B with no cutting plane and presolve
    """

    # Step 0: initialize some quantities
    n_knapsacks = len(capacities)
    n_items = len(profits)

    # Initialize the model
    model = Model("Knapsack")

    # Configure simple Branch-and-Bound
    model.setIntParam("presolving/maxrounds", 0)  # Disable presolve
    model.setIntParam("separating/maxrounds", 0)  # Disable cutting planes
    model.setHeuristics(SCIP_PARAMSETTING.OFF)  # Disable heuristics
    model.setIntParam("presolving/maxrestarts", 0)  # Disable restarts
    model.setIntParam("propagating/maxrounds", 0)  # Disable propagation

    x = dict()
    for j in range(n_items):
        for i in range(n_knapsacks):
            # Binary variables
            x[(j, i)] = model.addVar(name=f"x_{(j, i)}", lb=0.0, ub=1.0, vtype="B")

    # Each item must be assigned to at most one knapsack
    for j in range(n_items):
        model.addCons(sum(x[(j, i)] for i in range(n_knapsacks)) <= 1)

    # Add capacity constraint: sum(weights[j] * x[j]) <= capacity[i]
    for i in range(n_knapsacks):
        model.addCons(sum(weights[j] * x[(j, i)] for j in range(n_items)) <= capacities[i])

    # Set the objective
    model.setObjective(sum(profits[j] * x[(j, i)] for i in range(n_knapsacks) for j in range(n_items)), sense="maximize")

    # Optimize the model
    model.optimize()

    # Extract the solution
    if model.getStatus() == "optimal":
        total_profit = model.getObjVal()
        solution = {(j, i): model.getVal(x[(j, i)]) for j in range(n_items) for i in range(n_knapsacks) if model.getVal(x[(j, i)]) > 0}
        return solution, total_profit, True
    else:
        return None, None, False
