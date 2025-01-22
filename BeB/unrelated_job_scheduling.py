from bounds.unrelated_job_scheduling import binary_search, linear_relaxation
from utils import is_integer_val, is_integer_sol
import itertools as it


class Node:
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
        Compare two nodes. We will use a min heapq. The smallest lower bound is the best node (the "smallest" node)
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
        return f"fixed = {self.fixed}"  # TODO we may want to change this


class BranchAndBound:
    def __init__(self, node_selection_strategy, lower_bound, branching_rule, rounding_rule, epsilon):
        self.GLB = float("-inf")
        self.GUB = float("inf")
        self.node_selection_strategy = node_selection_strategy  # ["lowest_lower_bound", "depth_first", "breadth_first"]
        self.lower_bound_strategy = lower_bound  # ["lin_relax", "bin_search"]
        self.branching_rule = branching_rule  # ["max_min_proc", "max_avg_proc", ...]
        self.rounding_rule = rounding_rule  # ["best_matching", "all_to_shortest"]
        self.epsilon = epsilon

        # Control on alpha
        assert 0 < self.epsilon <= 1, "Alpha must be between 0 (<) and 1 (= 1 == Exact B&B)"

        # This will be instantiated in the solve method
        self.processing_times = None
        self.n_machines = None
        self.n_jobs = None
        self.GUB_argmin = None
        self.verbose = None
        self.TOL = None
        self.MAX_NODES = None

    def lower_bound(self, processing_times, initial_makespan, fixed):
        """Compute the lower bound for the problem with the specified parameters"""
        # Returns (X_frac, value, solvable)
        if self.lower_bound_strategy == "lin_relax":
            return linear_relaxation(processing_times, initial_makespan, fixed)
        if self.lower_bound_strategy == "bin_search":
            return binary_search(processing_times, initial_makespan, fixed)

    def branching_variable(self, X_frac):
        fractional_jobs = [j for (j, i) in X_frac.keys() if not is_integer_val(X_frac[(j, i)])]
        fractional_jobs = list(set(fractional_jobs))
        assert len(
            fractional_jobs) <= self.n_machines, "The number of fractional jobs is greater than the number of machines"
        if self.branching_rule == "max_min_proc":
            return max(fractional_jobs, key=lambda j: min(self.processing_times[j][i] for i in range(self.n_machines)))
        if self.branching_rule == "max_avg_proc":
            return max(fractional_jobs, key=lambda j:
            sum(self.processing_times[j][i] for i in range(self.n_machines)) / self.n_machines)

    def rounding(self, X_frac, fixed):
        # Returns (X_int, makespan) with in the form of a dict: X[(j,i)] = 1 if job j is assigned to machine i.
        # It does not include the fixed jobs !!!

        fractional_jobs = [j for (j, i) in X_frac.keys() if not is_integer_val(X_frac[(j, i)])]
        fractional_jobs = list(set(fractional_jobs))
        assert len(
            fractional_jobs) <= self.n_machines, "The number of fractional jobs is greater than the number of machines"

        # Integrally assigned jobs remain there
        X_int = {k: v for k, v in X_frac.items() if is_integer_val(v) and k not in fixed}

        completion_times = [0] * self.n_machines

        # Completion time of integrally assigned jobs.
        for (j, i) in X_int.keys():
            if abs(X_int[(j, i)] - 1) < self.TOL:  # <==> X_int[(j,i)] == 1
                completion_times[i] += self.processing_times[j][i]

        # Completion time of fixed jobs.
        for (j, i) in fixed:
            completion_times[i] += self.processing_times[j][i]

        if self.rounding_rule == "all_to_shortest":
            # Each fractional job is put on the machine where its processing time is minimal.
            for j in fractional_jobs:
                i = min([k for k in range(self.n_machines)], key=lambda x: self.processing_times[j][x])
                X_int[(j, i)] = 1
                completion_times[i] += self.processing_times[j][i]

            return X_int, max(completion_times)

        if self.rounding_rule == "best_matching":
            # The at most m fractional jobs are assigned in a matching such that the total makespan is minimal.
            # We iterate through all possible placements of the (<=m) fractional jobs on the m machines.

            best = None  # (permutation, makespan) for the best matching found so far
            for p in it.permutations(range(self.n_machines), len(fractional_jobs)):
                temp_comp_times = completion_times.copy()
                for ind in range(len(fractional_jobs)):
                    j = fractional_jobs[ind]
                    i = p[ind]
                    temp_comp_times[i] += self.processing_times[j][i]
                makespan = max(temp_comp_times)
                if best:
                    if makespan < best[1]:
                        best = p, makespan
                else:
                    best = p, makespan

            # Returning the best assignment
            p, makespan = best
            for ind in range(len(fractional_jobs)):
                j = fractional_jobs[ind]
                i = p[ind]
                X_int[(j, i)] = 1

            return X_int, makespan

    def stopping_criterion(self):
        return self.GUB / self.GLB < 1 + self.epsilon

    def solve(self, processing_times, verbose=0, opt=None):
        # Save the data
        self.processing_times = processing_times
        self.n_machines = len([processing_times[0]])  # All have the same length
        self.n_jobs = len(processing_times)

        self.GLB = float("-inf")
        self.GUB = float("inf")
        self.GUB_argmin = None  # Minimization --> Upper bound --> Heuristic --> This is integer

        self.verbose = verbose
        self.TOL = 1e-6
        self.MAX_NODES = 1e4

        pass
