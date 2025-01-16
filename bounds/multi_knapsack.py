from pyscipopt import Model


def dantzig_upper_bound(profits, weights, capacities, fixed, verbose=False):
    """
    Compute the upper bound of the multi knapsack using the algorithm of George Dantzig (Discrete variable extremum points, 1957)
    - The capacities are already reduced
    """
    # Step 0: initialize some quantities
    n_knapsacks = len(capacities)
    n_items = len(profits)

    temp_capacities, temp_weights = capacities.copy(), weights.copy()

    # Step 1: get the not fixed items
    fixed_items = []
    for (j, i) in fixed:
        fixed_items.append(j)

    # Items are sorted in the main
    sorted_items = [j for j in range(n_items) if j not in fixed_items]

    X_frac = {}

    while max(temp_capacities) > 0 and len(sorted_items) > 0:
        # Pop the first item from the list
        j_to_fix = sorted_items.pop(0)
        starting_weight = temp_weights[j_to_fix]

        already_assigned_partially = False

        # Create a list where the item r sum up the capacities until r
        capacities_sum = [sum(temp_capacities[:r + 1]) for r in range(len(temp_capacities))]

        # Get the minimum index for with weights[j_to_fix] is smaller than capacities sum
        r_list = [r for r in range(n_knapsacks) if capacities_sum[r] >= temp_weights[j_to_fix]]
        if len(r_list) > 0:
            r_min = min(r_list)

            for i in range(r_min + 1):
                if starting_weight <= temp_capacities[i] and not already_assigned_partially:
                    # Assign it at the most (integrally)
                    X_frac[(j_to_fix, i)] = 1
                    temp_capacities[i] = temp_capacities[i] - starting_weight
                    temp_weights[j_to_fix] = temp_weights[j_to_fix] - starting_weight  # Done
                else:
                    q = min(temp_capacities[i], temp_weights[j_to_fix]) / starting_weight
                    X_frac[(j_to_fix, i)] = q
                    temp_capacities[i] = temp_capacities[i] - q * starting_weight
                    temp_weights[j_to_fix] = temp_weights[j_to_fix] - q * starting_weight
                    already_assigned_partially = True

        else:
            # You have done! Just fit the last element
            while max(temp_capacities) > 0:
                for i in range(n_knapsacks):
                    if temp_capacities[i] > 0:
                        q = temp_capacities[i] / temp_weights[j_to_fix]
                        X_frac[(j_to_fix, i)] = q
                        temp_capacities[i] = 0
        if verbose:
            print(X_frac)

    total_profit = sum([X_frac[(j, i)] * profits[j] for (j, i) in X_frac.keys()])
    return X_frac, total_profit, True


def dantzig_upper_bound_linear_relaxation(profits, weights, capacities, fixed):
    """
    Compute the upper bound of the multi knapsack using the algorithm of George Dantzig (Discrete variable extremum points, 1957)
    """

    # Step 0: initialize some quantities
    n_knapsacks = len(capacities)
    n_items = len(profits)

    # Step 1: get the not fixed items # TODO sort the item at the very beginning
    fixed_items = []
    for (j, i) in fixed:
        fixed_items.append(j)

    not_fixed_items = [j for j in range(n_items) if j not in fixed_items]
    knapsacks_with_non_zero_capacity = [i for i in range(n_knapsacks) if capacities[i] > 0]

    # Initialize the model
    model = Model("Knapsack")

    model.hideOutput()

    x = dict()
    for j in not_fixed_items:
        for i in knapsacks_with_non_zero_capacity:
            x[(j, i)] = model.addVar(name=f"x_{(j, i)}", lb=0.0, ub=1.0, vtype="C")  # Continuous variables for linear relaxation

    # Each item must be assigned to at most one knapsack
    for j in not_fixed_items:
        model.addCons(sum(x[(j, i)] for i in knapsacks_with_non_zero_capacity) <= 1)

    # Add capacity constraint: sum(weights[j] * x[j]) <= capacity[i]
    for i in knapsacks_with_non_zero_capacity:
        try:
            model.addCons(sum(weights[j] * x[(j, i)] for j in not_fixed_items) <= capacities[i])
        except:
            print(" ")

    # # For the fixed items, just add an extra constrain
    # for (j, i) in fixed:
    #     if i < n_knapsacks:
    #         model.addCons(x[(j, i)] == 1)
    #     else:
    #         for i in range(n_knapsacks):
    #             model.addCons(x[(j, i)] == 0)

    # Set the objective
    model.setObjective(sum(profits[j] * x[(j, i)] for i in knapsacks_with_non_zero_capacity for j in not_fixed_items), sense="maximize")

    # Optimize the model
    model.optimize()

    # Extract the solution
    if model.getStatus() == "optimal":
        total_profit = model.getObjVal()
        solution = {(j, i): model.getVal(x[(j, i)]) for j in not_fixed_items for i in knapsacks_with_non_zero_capacity if model.getVal(x[(j, i)]) > 0}
        return solution, total_profit, True
    else:
        return None, None, False
