from bounds.multi_knapsack import dantzig_upper_bound
from utils import is_integer_val
from itertools import combinations


class Node():
    def __init__(self, UB, LB, depth, strategy, fixed = []):
        self.UB = UB
        self.strategy = strategy
        self.depth = depth
        self.fixed = fixed

    def __le__(self, other):
        """
        Compare two nodes. We will use a min heapq. Lo greatest UB is the best node (the "smallest" node)
        """
        if self.strategy == "greatest_upper_bound":
            return self.UB >= other.UB
        elif self.strategy == "depth_first":
            return self.depth >= other.depth
        elif self.strategy == "breadth_first":
            return self.depth <= other.depth

class BranchAndBound():
    def __init__(self, node_selction_strategy, upper_bound, branching_rule, rounding_rule):
        self.GLB = float("-inf")
        self.GUB = float("inf")
        self.node_selection_strategy = node_selction_strategy
        self.upper_bound_strategy = upper_bound
        self.branching_rule_strategy = branching_rule
        self.rounding_rule_strategy = rounding_rule

        # This will be instantiated in the solve method
        self.profits = None
        self.weights = None
        self.capacities = None
        self.n_knapsacks = None


    def upper_bound(self, profits, weights, capacities, fixed):
        if self.upper_bound_strategy == "dantzig_upper_bound":
            return dantzig_upper_bound(profits, weights, capacities, fixed)

    def branching_variable(self, X_frac):
        if self.branching_rule_strategy == "critical_element":
            # Get all the fractional items
            fractional_items = [j for (j, i) in X_frac.keys() if not is_integer_val(X_frac[(j, i)])]
            fractional_items = list(set(fractional_items))
            assert len(fractional_items) <= self.n_knapsacks, "The number of fractional items is greater than the number of knapsacks"
            return max(fractional_items, key=lambda j: self.profits[j])

    def rounding(self, X_frac):
        if self.rounding_rule_strategy == "martello_toth_rule":
            fractional_items = [j for (j, i) in X_frac.keys() if not is_integer_val(X_frac[(j, i)])]
            # Create all the combinations that assign each fractional item to a knapsack
            # The key is the item, the value is the knapsack
            possible_assignments = {}




    def solve(self, profits, weights, capacities):
        # Save the data
        self.profits = profits
        self.weights = weights
        self.capacities = capacities
        self.n_knapsacks= len(self.capacities)

        # Istantiate the root node
        X_frac, total_profit = self.upper_bound(profits, weights, capacities, [])

