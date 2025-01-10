from bounds.multi_knapsack import dantzig_upper_bound
from utils import is_integer_val, is_integer_sol
import time
from heapq import heappush, heappop

class Node():
    def __init__(self, X_frac, UB, depth, strategy, fixed, capacities):
        self.UB = UB
        self.X_frac = X_frac
        self.strategy = strategy
        self.depth = depth
        self.fixed = fixed
        self.capacities = capacities

        # This is updated in the solve method
        self.LB = None
        self.X_int = None

    def __lt__(self, other):
        """
        Compare two nodes. We will use a min heapq. Lo greatest UB is the best node (the "smallest" node)
        """
        if self.strategy == "greatest_upper_bound":
            return self.UB >= other.UB
        elif self.strategy == "depth_first":
            return self.depth >= other.depth
        elif self.strategy == "breadth_first":
            return self.depth <= other.depth

    def update(self, X_int, LB):
        self.LB = LB
        self.X_int = X_int

class BranchAndBound():
    def __init__(self, node_selction_strategy, upper_bound, branching_rule, rounding_rule, alpha):
        self.GLB = float("-inf")
        self.GUB = float("inf")
        self.node_selection_strategy = node_selction_strategy
        self.upper_bound_strategy = upper_bound
        self.branching_rule_strategy = branching_rule
        self.rounding_rule_strategy = rounding_rule
        self.alpha = alpha

        # Control on alpha
        assert 0 < self.alpha <= 1, "Alpha must be between 0 (<) and 1 (= 1 == Exact B&B)"

        # This will be instantiated in the solve method
        self.profits = None
        self.weights = None
        self.capacities = None
        self.n_knapsacks = None


    def upper_bound(self, profits, weights, capacities, fixed):
        """
        Compute the upper bound for the multi-knapsack problem using the specified strategy.

        Parameters:
        - profits: List of profits for each item.
        - weights: List of weights for each item.
        - capacities: List of capacities for each knapsack.
        - fixed: List of fixed item assignments (item, knapsack).

        Returns:
        - X_frac: Fractional solution for the upper bound.
        - total_profit: Total profit of the fractional solution.
        - True: Indicates the solution is feasible.
        """
        if self.upper_bound_strategy == "dantzig_upper_bound":
            return dantzig_upper_bound(profits, weights, capacities, fixed)

    def branching_variable(self, X_frac):
        # Get all the fractional items
        fractional_items = [j for (j, i) in X_frac.keys() if not is_integer_val(X_frac[(j, i)])]
        fractional_items = list(set(fractional_items))
        assert len(fractional_items) <= self.n_knapsacks, "The number of fractional items is greater than the number of knapsacks"
        if self.branching_rule_strategy == "critical_element":
            # Branch on the item with the highest profit
            return max(fractional_items, key=lambda j: self.profits[j])
        if self.branching_rule_strategy == "profit_per_weight_ratio":
            # Branch on the item with the highest ratio profit/weight
            return max(fractional_items, key=lambda j: self.profits[j]/self.weights[j])

    def rounding(self, X_frac, capacities, new_fixed):
        """
        This is with the REDUCED capacities
        """
        if self.rounding_rule_strategy == "martello_toth_rule":
            fractional_items = [j for (j, i) in X_frac.keys() if not is_integer_val(X_frac[(j, i)])]
            candidate_solutions = []

            # Candidate solution 1: remove all the fractional items AND the item that are naturally fixed in the path
            X_int = {k: v for k, v in X_frac.items() if is_integer_val(v) and k not in new_fixed}

            total_profit = sum([self.profits[j] * X_int[(j, i)] for (j, i) in X_int.keys()])
            candidate_solutions.append((X_int, total_profit))

            # Candidate solution 2, 3, ..., m + 1
            for j in fractional_items:
                for i in range(self.n_knapsacks):
                    if self.weights[j] <= capacities[i]:
                        X_int = {(j, i) : 1}
                        candidate_solutions.append((X_int, self.profits[j]))
                        break

            # Return the best candidate solution
            return max(candidate_solutions, key=lambda x: x[1]) # This is just partial!

    def stopping_criterion(self):
        if self.alpha == 1:
            return self.GUB - self.GLB >= 1 # Enough because weights an profits are integer
        else:
            return self.GLB / self.GUB < self.alpha



    def solve(self, profits, weights, capacities, verbose = 0):

        # Save the data
        self.profits = profits
        self.weights = weights
        self.global_capacities = capacities
        self.n_knapsacks= len(self.global_capacities)

        self.GLB = float("-inf")
        self.GUB = float("inf")
        self.GLB_argmin = None # Maximiazion --> Lowerbound --> Heuristic --> This is integer

        self.verbose = verbose

        start = time.time()

        # Istantiate the root node
        depth = 0
        X_frac, UB, feas = self.upper_bound(profits, weights, capacities.copy(), [])
        self.GUB = UB

        # If X_frac is integer, we have a feasible solution, and return
        if is_integer_sol(X_frac):
            self.GLB = UB
            self.GLB_argmin = X_frac
            if verbose >= 1:
                print("UB = ", UB, "LB = ", UB, "--> Solved at the root node", flush = True)
            return UB, X_frac, time.time() - start, depth

        # If this is not the case, round the solution
        X_int, LB = self.rounding(X_frac, self.global_capacities, [])
        # X_frac, UB, depth, strategy, fixed, capacities
        root_node = Node(X_frac, UB, depth, self.node_selection_strategy, [], self.global_capacities)
        root_node.update(X_int, LB)

        # Update the global lower bound
        self.GLB = max(self.GLB, LB)
        self.GLB_argmin = X_int


        # Initialize a min heapq
        queue = []
        heappush(queue, root_node)

        if verbose >= 0.5:
            print("Root node: UB = ", UB, "LB = ", LB, flush = True)

        nodes_explored = 1
        while self.stopping_criterion():
            parent_node = heappop(queue) # Get the node with the highest UB
            if self.node_selection_strategy == "greatest_upper_bound":
                self.GUB = parent_node.UB
            nodes_explored += 1
            if verbose >= 1:
                print(f"Exploring node {nodes_explored - 1}")
                print(f"Node UB: {parent_node.UB}, Node LB: {parent_node.LB}, path of the node: {parent_node.fixed}")

            # Branching
            j = self.branching_variable(parent_node.X_frac) # Node 12 issue :O

            capacities_parent_node =  parent_node.capacities.copy()

            for i in range(self.n_knapsacks + 1):
                if verbose >= 1:
                    print(f"Fixing job {j} on knapsack {i}")
                # Fix the item j on the knapsack i
                new_fixed = parent_node.fixed + [(j, i)]

                # Get the capacities
                new_capacities = capacities_parent_node.copy()
                if i < self.n_knapsacks:
                    new_capacities[i] -= weights[j]


                if min(new_capacities) >= 0: # Else --> Prune by infeasibility
                    new_capacities_keep = new_capacities.copy()
                    X_frac, UB, feas = self.upper_bound(profits.copy(), weights.copy(), new_capacities, new_fixed)

                    if feas: # Else --> Prune by infeasibility
                        # Add the fixed to X_frac
                        for (j, i) in new_fixed:
                            if i < self.n_knapsacks:
                                X_frac[(j, i)] = 1
                                UB += profits[j]

                        if is_integer_sol(X_frac):
                            if UB > self.GLB:
                                self.GLB = UB
                                self.GLB_argmin = X_frac
                            # Else, prune by integrality, so don't do anything
                        else:
                            # Do the rounding!
                            X_int, LB = self.rounding(X_frac, new_capacities_keep, new_fixed) # This is just partial, now you have to complete everything with the fixed
                            for (j, i) in new_fixed:
                                if i < self.n_knapsacks:
                                    X_int[(j, i)] = 1
                                    LB += profits[j]


                            # if verbose >= 1:
                            #     print(f"\twith new capacities {new_capacities_keep} --> UB = {UB}, LB = {LB}")

                            # Now we have everything in place, do we add this node?
                            if UB >= self.GLB:
                                # Add it to the queue
                                # X_frac, UB, depth, strategy, fixed, capacities
                                node = Node(X_frac, UB, parent_node.depth + 1, self.node_selection_strategy, new_fixed, new_capacities_keep)
                                node.update(X_int, LB)
                                # If LB = UB, don't add it!
                                if node.UB - node.LB > 1:
                                    heappush(queue, node)

                                if verbose >= 1:
                                    print("\tNode added to the queue")

                                # If also the lowerbound is higher:
                                if LB > self.GLB:
                                    if verbose >= 0.5:
                                        old_lb = self.GLB
                                        print(f"!!! Improved global lowerbound: {old_lb} --> {LB}")
                                    self.GLB = LB
                                    self.GLB_argmin = X_int
                            else:
                                if verbose >= 1:
                                    print("\tPruned by bound")
                    else:
                        if verbose >= 1:
                            print("\tPruned by infeasibility (linear program infeasible)")
                else:
                    if verbose >= 1:
                        print("\tPruned by infeasibilty (negative capacity)")
                #print(" ")



        return self.GLB, X_int, self.GUB, time.time() - start


