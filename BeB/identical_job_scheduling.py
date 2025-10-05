from bounds.unrelated_job_scheduling import binary_search, linear_relaxation
from utils import is_integer_val, is_integer_sol
import itertools as it
import time
from heapq import heappush, heappop


class Node:
    def __init__(self, X_frac, LB, depth, strategy, fixed, overhead):
        self.LB = LB
        self.X_frac = X_frac
        self.strategy = strategy
        self.depth = depth
        self.fixed = fixed
        self.overhead = overhead  # Vector of length n_machines

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
        self.LLB = float("inf")
        self.LUB = float("-inf")
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
        self.LUB_argmin = None
        self.verbose = None
        self.TOL = None
        self.MAX_NODES = None

    def lower_bound(self, processing_times, overhead, fixed):
        """Compute the lower bound for the problem with the specified parameters"""
        # Returns (X_frac, value, solvable)
        if self.lower_bound_strategy == "lin_relax":
            return linear_relaxation(processing_times, overhead, fixed)
        if self.lower_bound_strategy == "bin_search":
            return binary_search(processing_times, overhead, fixed)

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
        return self.LUB / self.LLB < 1 + self.epsilon

    def solve(self, processing_times, verbose=0, opt=None):
        # Save the data
        self.processing_times = processing_times
        self.n_machines = len(processing_times[0])  # All have the same length
        self.n_jobs = len(processing_times)

        self.LLB = float("-inf")
        self.LUB = float("inf")
        self.LUB_argmin = None  # Minimization --> Upper bound --> Heuristic --> This is integer

        self.verbose = verbose
        self.TOL = 1e-6
        self.MAX_NODES = 1e4

        start = time.time()

        # Instantiate the root node
        depth = 0
        X_frac, LB, feas = self.lower_bound(self.processing_times, [0]*self.n_machines, [])
        self.LLB = LB

        # If X_frac is integer, we have an optimal solution and return
        if is_integer_sol(X_frac):
            self.LUB = LB
            self.LUB_argmin = X_frac
            if verbose >= 1:
                print("LB = ", LB, "UB = ", LB, "--> Solved at the root node", flush=True)
            return LB, X_frac, LB, time.time() - start, 0, 0, 0, True

        # If this is not the case, we round the solution
        X_int, UB = self.rounding(X_frac, [])
        root_node = Node(X_frac, LB, depth, self.node_selection_strategy, [], [0]*self.n_machines)
        root_node.update(X_int, UB)

        # Update the global lower bound
        self.LUB = UB
        self.LUB_argmin = X_int

        # Initialize a min heapq
        queue = []
        heappush(queue, root_node)

        if verbose >= 0.5:
            print("Root node: UB = ", UB, "LB = ", LB, flush=True)

        nodes_explored = 1  # Number of nodes explored
        max_depth = 0
        nodes_opt = -1  # No optimal solution
        not_yet_opt = True  # Of course, it's not optimal...

        while queue:
            # Get the next node
            parent_node = heappop(queue)

            # Update the max_depth if needed
            max_depth = max(max_depth, len(parent_node.fixed))
            nodes_explored += 1

            if verbose >= 2:
                print(f"Exploring node {nodes_explored - 1}")
                print(f"Node LB: {parent_node.LB}, Node UB: {parent_node.UB}")
                print(f"path of the node: {parent_node.fixed}")

            # We create (at most) m+1 children.
            j = self.branching_variable(parent_node.X_frac)

            for q in range(self.n_machines):
                if verbose >= 2:
                    print(f"Fixing job {j} on machine {q}")

                # Get the overheads
                new_overhead = parent_node.overhead.copy()

                # Update the corresponding overhead of the children
                new_overhead[q] += self.processing_times[j][q]

                # Fix item j on machine q
                new_fixed = parent_node.fixed + [(j, q)]

                # If there are no more free jobs: we calculate the solution, compare it with the best
                # and prune the branch.
                if len(new_fixed) == self.n_jobs:
                    if verbose >= 2:
                        print("X_frac is empty, all the items are fixed")
                    UB = max(new_overhead)
                    if UB < self.LUB:
                        if verbose >= 2:
                            print(f"\t!!! Improved global upper bound: {self.LUB} --> {UB} (by integrality)")
                        self.LUB = UB
                        self.LUB_argmin = {k: 1 for k in new_fixed}
                    continue

                # If we are here, there are still some free jobs left.
                # We round up the fractional optimum, and add back fixed items.
                X_frac, LB, _ = self.lower_bound(processing_times, new_overhead, new_fixed)

                # Add the fixed to X_frac
                for (k, i) in new_fixed:
                    X_frac[(k, i)] = 1

                # If the lower bound is worse than self.LUB, then we prune
                if LB >= self.LUB:
                    if verbose >= 2:
                        print("\tPruned by bound")
                    continue

                # If we are here, LB <= self.LUB. If X_frac is integer, we have found a better solution.
                if is_integer_sol(X_frac):
                    if verbose >= 2:
                        print(f"\t!!! Improved global upper bound: {self.LUB} --> {LB} (by integrality)")
                    self.LUB = LB
                    self.LUB_argmin = X_frac
                    continue

                # Else, we do the rounding!
                X_int, UB = self.rounding(X_frac, new_fixed)

                # This is just partial, now you have to complete everything with the fixed jobs
                for (k, i) in new_fixed:
                    X_int[(k, i)] = 1
                if verbose >= 2:
                    print(f"\t Rounding done: LB = {LB}, UB = {UB}, X_frac = {X_frac}")

                # Add the node to the queue
                node = Node(X_frac, LB, parent_node.depth + 1, self.node_selection_strategy, new_fixed, new_overhead)
                node.update(X_int, UB)
                if verbose >= 2:
                    print("\t" + str(node))
                heappush(queue, node)

                # If also the upper bound is lower:
                if UB < self.LUB:
                    if verbose >= 0.5:
                        old_ub = self.LUB
                        print(f"!!! Improved global upper bound: {old_ub} --> {UB}")
                    self.LUB = UB
                    self.LUB_argmin = X_int

            """
            We have processed all m+1 children. The ones that are relevant are put back in the queue.
            The queue now contains a complete partition of the solution space, except for nodes
            pruned by integrality.
            We update LLB as the minimal lower bound in the queue, if it's not empty. Else, it is 
            simply self.LUB.
            """
            if verbose >= 2:
                print("Nodes explored:", nodes_explored)
                print("Queue length = ", len(queue))

            # It might happen that min(node.LB for node in queue) is not "real LLB", if it is achieved by
            # an integral node that we did not add to the queue. Then, the real LLB is LUB.
            self.LLB = min(self.LUB, min(node.LB for node in queue) if queue else self.LUB)
            if verbose >= 2:
                print(f"Current lower bound: {self.LLB}")

            """
            At each iteration, check if we have actually reached the optimal solution.
            """
            if abs(self.LUB - opt) <= self.TOL and not_yet_opt:
                nodes_opt = nodes_explored
                not_yet_opt = False

            # We stop if the stopping criterion holds, or if we explored too many nodes.
            if nodes_explored > self.MAX_NODES:
                if not_yet_opt:
                    nodes_opt = nodes_explored
                return self.LUB, self.LUB_argmin, self.LLB, time.time() - start, nodes_explored, nodes_opt, max_depth, False  # Terminating because of the number of nodes

            if self.stopping_criterion():
                if not_yet_opt:
                    nodes_opt = nodes_explored
                return self.LUB, self.LUB_argmin, self.LLB, time.time() - start, nodes_explored, nodes_opt, max_depth, True  # Terminating because of the stopping criterion

        """ 
        At this point, the queue is empty. 
        There is no more lower bound, so the best integer solution (self.LUB) is optimal.
        """
        nodes_opt = nodes_explored
        return self.LUB, self.LUB_argmin, self.LLB, time.time() - start, nodes_explored, nodes_opt, max_depth, True
