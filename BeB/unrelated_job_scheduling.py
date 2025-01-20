class Node():
    def __init__(self, X_frac, LB, depth, strategy, fixed):
        self.LB = LB
        self.X_frac = X_frac
        self.strategy = strategy
        self.depth = depth
        self.fixed = fixed

        # This is updated in the solve method
        self.UB = None
        self.X_int = None

    def __lt__(self, other):
        """
        Compare two nodes. We will use a min heapq. The smallest lowerbound is the best node (the "smallest" node)
        """
        if self.strategy == "lowest_lower_bound":
            return self.LB <= other.LB
        elif self.strategy == "depth_first":
            return self.depth >= other.depth
        elif self.strategy == "breadth_first":
            return self.depth <= other.depth

    def update(self, X_int, UB):
        self.UB = UB
        self.X_int = X_int

    def __str__(self):
        return f"fixed = {self.fixed}" # TODO we may want to change this


class BranchAndBound():
    def __init__(self, node_selection_strategy, lower_bound, branching_rule, rounding_rule, epsilon):
        self.GLB = float("-inf")
        self.GUB = float("inf")
        self.node_selection_strategy = node_selection_strategy
        self.lower_bound_strategy = lower_bound
        self.branching_rule_strategy = branching_rule
        self.rounding_rule_strategy = rounding_rule
        self.epsilon = epsilon

        # Control on alpha
        assert 0 < self.epsilon <= 1, "Alpha must be between 0 (<) and 1 (= 1 == Exact B&B)"

        # This will be instantiated in the solve method
        self.completion_times = None
        self.n_machines = None
        self.n_jobs = None
        self.GUB_argmin = None
        self.verbose = None
        self.TOL = None
        self.MAX_NODES = None

    def lower_bound(self, completion_times, fixed):
        pass

    def branching_variable(self, X_frac, node):
        pass

    def rounding(self, X_frac):
        pass

    def stopping_criterion(self):
        return self.GUB / self.GLB < 1 + self.epsilon

    def solve(self, completion_times, verbose=0, opt=None):
        # Save the data
        self.completion_times = completion_times

        self.n_machines = len([completion_times[0]]) # All of the same length
        self.n_jobs = len(completion_times)

        self.GLB = float("-inf")
        self.GUB = float("inf")
        self.GUB_argmin = None  # Minimization --> Upper bound --> Heuristic --> This is integer

        self.verbose = verbose
        self.TOL = 1e-6
        self.MAX_NODES = 1e4

        pass