from ortools.linear_solver import pywraplp


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
