from pyscipopt import Model

def dantzig_upper_bound(profits, weights, capacities, fixed):
    """
    Compute the upper bound of the multi knapsack using the algorithm of George Dantzig (Discrete variable extremum points, 1957)
    """
    n_knapsacks = len(capacities)
    n_items = len(profits)

    # Prepocessing: the fixed items (j, i) are item j fixed on knapsack i. We will reduce the capacities of the knapsacks acordingly.
    fixed_items = []
    for (j, i) in fixed:
        weights[j] = 0
        profits[j] = 0
        fixed_items.append(j)

    # Sort the elements decreasingly
    not_fixed_items = [j for j in range(n_items) if j not in fixed_items]

    sorted_items = {}
    for j in not_fixed_items:
        sorted_items[j] = profits[j] / weights[j]

    sorted_items = dict(sorted(sorted_items.items(), key=lambda item: item[1], reverse=True))

    current_knapsack = 0
    current_item_idx = 0
    sorted_items = list(sorted_items.keys())

    X_frac = {}



    while current_item_idx < len(not_fixed_items) and current_knapsack < n_knapsacks:

        # If the sum of the capacities is smaller than the first element to be stored, store it separatetly and return
        if sum(capacities) < weights[sorted_items[current_item_idx]]:
            for i in range(n_knapsacks):
                if capacities[i] > 0:
                    X_frac[(sorted_items[current_item_idx], i)] = capacities[i] / weights[sorted_items[current_item_idx]]
                    capacities[i] = 0
            total_profit = sum([profits[j] * X_frac[(j, i)] for (j, i) in X_frac.keys()])
            return X_frac, total_profit, True

        if capacities[current_knapsack] == 0:
            current_knapsack += 1
        else:
            if weights[sorted_items[current_item_idx]] <= capacities[current_knapsack]:
                # If it fits integrally, fit it integrally
                X_frac[(sorted_items[current_item_idx], current_knapsack)] = 1
                capacities[current_knapsack] -= weights[sorted_items[current_item_idx]]
                current_item_idx += 1
            else:
                # Otherwise, fit it fractionally
                q = capacities[current_knapsack] / weights[sorted_items[current_item_idx]]
                X_frac[(sorted_items[current_item_idx], current_knapsack)] = q

                # Zero the capacity of the knapsack
                capacities[current_knapsack] = 0

                # Move to the next knapsack
                current_knapsack += 1

                if current_knapsack < n_knapsacks:

                    # Fit the remaining part of the item
                    X_frac[(sorted_items[current_item_idx], current_knapsack)] = 1 - q
                    capacities[current_knapsack] -= weights[sorted_items[current_item_idx]] * (1 - q)

                    # Increment the item
                    current_item_idx += 1
        if max(capacities) == 0:
            # If you have saturated
            current_item_idx = len(fixed_items)
            current_knapsack = n_knapsacks

    total_profit = sum([profits[j] * X_frac[(j, i)] for (j, i) in X_frac.keys()])
    return X_frac, total_profit, True


def dantzig_upper_bound_linear_relaxation(profits, weights, capacities, fixed):
    """
    Compute the upper bound of the multi knapsack using the algorithm of George Dantzig (Discrete variable extremum points, 1957)
    """
    n_knapsacks = len(capacities)
    n_items = len(profits)

    # Prepocessing: the fixed items (j, i) are item j fixed on knapsack i. We will reduce the capacities of the knapsacks acordingly.
    for (j, i) in fixed:
        if i < n_knapsacks:
            weights[j] = 0
            profits[j] = 0

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




