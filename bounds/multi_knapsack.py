from pyscipopt import Model


def dantzig_upper_bound(profits, weights, capacities, fixed, verbose = False):
    """
    Compute the upper bound of the multi knapsack using the algorithm of George Dantzig (Discrete variable extremum points, 1957)
    - The capacities are already reduced
    """
    # Step 0: initialize some quantities
    n_knapsacks = len(capacities)
    n_items = len(profits)

    # Step 1: get the not fixed items
    fixed_items = []
    for (j, i) in fixed:
        fixed_items.append(j)

    not_fixed_items = [j for j in range(n_items) if j not in fixed_items]

    # Step 2: sort the item
    sorted_items = {}
    for j in not_fixed_items:
        sorted_items[j] = profits[j] / weights[j]

    sorted_items = dict(sorted(sorted_items.items(), key=lambda item: item[1], reverse=True))

    # Define some variables you may need
    sorted_items = list(sorted_items)
    X_frac = {}

    while max(capacities) > 0 and len(sorted_items) > 0:
        # Pop the first item from the list
        j_to_fix = sorted_items.pop(0)
        starting_weight = weights[j_to_fix]

        already_assigned_partially = False

        # Create a list where the item r sum up the capacities until r
        capacities_sum = [sum(capacities[:r + 1]) for r in range(len(capacities))]

        # Get the minimum index for with weights[j_to_fix] is smaller than capacities sum
        r_list = [r for r in range(n_knapsacks) if capacities_sum[r] >= weights[j_to_fix]]
        if len(r_list) > 0:
            r_min = min(r_list)

            for i in range(r_min + 1):
                if starting_weight <= capacities[i] and not already_assigned_partially:
                    # Assign it at the most (integrally)
                    X_frac[(j_to_fix, i)] = 1
                    capacities[i] = capacities[i] -  starting_weight
                    weights[j_to_fix] = weights[j_to_fix] -  starting_weight # Done
                else:
                    q = min(capacities[i], weights[j_to_fix]) / starting_weight
                    X_frac[(j_to_fix, i)] = q
                    capacities[i] = capacities[i] - q * starting_weight
                    weights[j_to_fix] = weights[j_to_fix] - q * starting_weight
                    already_assigned_partially = True

        else:
            # You have done! Just fit the last element
            while max(capacities) > 0:
                for i in range(n_knapsacks):
                    if capacities[i] > 0:
                        q = capacities[i] / weights[j_to_fix]
                        X_frac[(j_to_fix, i)] = q
                        capacities[i] = 0
        if verbose:
            print(X_frac)

    total_profit = sum([X_frac[(j, i)] * profits[j] for (j, i) in X_frac.keys()])
    return X_frac, total_profit, True


def dantzig_upper_bound_linear_relaxation(profits, weights, capacities, fixed):
    """
    Compute the upper bound of the multi knapsack using the algorithm of George Dantzig (Discrete variable extremum points, 1957)
    """
    n_knapsacks = len(capacities)
    n_items = len(profits)

    # Initialize the model
    model = Model("Knapsack")

    model.hideOutput()

    x = dict()
    for j in range(n_items):
        for i in range(n_knapsacks):
            x[(j, i)] = model.addVar(name=f"x_{(j, i)}", lb=0.0, ub=1.0, vtype="C")  # Continuous variables for linear relaxation

    # Each item must be assigned to at most one knapsack
    for j in range(n_items):
        model.addCons(sum(x[(j, i)] for i in range(n_knapsacks)) <= 1)

    # Add capacity constraint: sum(weights[j] * x[j]) <= capacity[i]
    for i in range(n_knapsacks):
        model.addCons(sum(weights[j] * x[(j, i)] for j in range(n_items)) <= capacities[i])

    # For the fixed items, just add an extra constrain
    for (j, i) in fixed:
        if i < n_knapsacks:
            model.addCons(x[(j, i)] == 1)
        else:
            model.addCons(x[(j, i)] == 0)

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




