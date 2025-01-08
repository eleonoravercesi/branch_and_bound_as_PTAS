def dantzig_upper_bound(profits, weights, capacities, fixed):
    """
    Compute the upper bound of the multi knapsack using the algorithm of George Dantzig (Discrete variable extremum points, 1957)
    """
    n_knapsacks = len(capacities)
    n_items = len(profits)

    # Prepocessing: the fixed items (j, i) are item j fixed on knapsack i. We will reduce the capacities of the knapsacks acordingly.
    for (j, i) in fixed:
        if i < n_knapsacks:
            capacities[i] -= weights[j]
            weights[j] = 0
            profits[j] = 0

    # Sort the elements decreasingly
    sorted_items = {}
    for j in range(n_items):
        if weights[j] == 0:
            sorted_items[j] = 0
        else:
            sorted_items[j] = profits[j] / weights[j]

    sorted_items = dict(sorted(sorted_items.items(), key=lambda item: item[1], reverse=True))

    current_knapsack = 0
    current_item = 0
    sorted_items = list(sorted_items.keys())

    X_frac = {}

    while current_item < n_items and current_knapsack < n_knapsacks:
        if weights[sorted_items[current_item]] <= capacities[current_knapsack]:
            # If it fits integrally, fit it integrally
            X_frac[(sorted_items[current_item], current_knapsack)] = 1
            capacities[current_knapsack] -= weights[sorted_items[current_item]]
            current_item += 1
        else:
            # Otherwise, fit it fractionally
            q = capacities[current_knapsack] / weights[sorted_items[current_item]]
            X_frac[(sorted_items[current_item], current_knapsack)] = q

            # Zero the capacity of the knapsack
            capacities[current_knapsack] = 0

            # Move to the next knapsack
            current_knapsack += 1

            if current_knapsack < n_knapsacks:

                # Fit the remaining part of the item
                X_frac[(sorted_items[current_item], current_knapsack)] = 1 - q
                capacities[current_knapsack] -= weights[sorted_items[current_item]] * (1 - q)

                # Increment the item
                current_item += 1

    total_profit = sum([profits[j] * X_frac[(j, i)] for (j, i) in X_frac.keys()])
    return X_frac, total_profit




