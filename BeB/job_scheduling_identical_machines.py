from lower_bounds.lower_bounds_JS_identical import linear_relaxation, binary_search
import heapq
from utils import is_integer_sol, is_integer_val
import time

class Node():
    """Class representing a subproblem node in the branch and bound tree"""
    def __init__(self, level, lower_bound, X_frac, fixed_vars = None, rounding_rule="arbitrary_rounding"):
        self.level = level
        self.lower_bound = lower_bound
        self.X_frac = X_frac
        self.rounding_rule = rounding_rule
        if fixed_vars is None:
            self.fixed_vars = [(0, 0)]
        else:
            self.fixed_vars = fixed_vars # You can always assign the first job to the first machine w.l.o.g.

        # Get the number of jobs
        self.n_jobs = max([j for (j, i) in X_frac.keys()]) + 1 # Every job must be assigned

        # This will be updated in the init
        self.X_int = None
        self.upper_bound = None # This will be set by the solver

        if rounding_rule == "arbitrary_rounding":
            # Just round each job to the machine with the largest fraction
            keys_not_integer = set()
            for (j, i) in X_frac.keys():  # job, machine
                if not abs(round(X_frac[j, i]) - X_frac[j, i]):
                    keys_not_integer.add(j)
            # Now I have a list with all the non integer jobs
            X_int = {}
            # TODO there is an error HERE (the solution is not correct)
            # Every job must be assigned to a machine
            for j in range(self.n_jobs):
                # Get all the assignation of this job
                assignations = [(j_prime, i) for (j_prime, i) in X_frac.keys() if j_prime == j]
                if len(assignations) == 1:
                    # Is integer if you assign it just one (not splitted)
                    X_int[assignations[0]] = 1
                else:
                    # Pick the assignation where is maximal
                    x = max(assignations, key=lambda x: X_frac[x])
                    X_int[x] = 1
            # Now I have the integer solution, do some cheks
            assert len(X_int) == self.n_jobs
            # Check that some jobs are not assigned twice
            assert len(set([j for (j, i) in X_int.keys()])) == self.n_jobs
            self.X_int = X_int

    def set_upper_bound(self, P, n_machines):
        upper_bound = [0 for _ in range(n_machines)]
        for (j, i) in self.X_int.keys():
            upper_bound[i] += P[j]
        self.upper_bound = max(upper_bound)

    def get_upper_bound(self):
        return self.upper_bound, self.X_int

    def get_fractional_solution(self):
        return self.X_frac

    def get_fixed_vars(self):
        return self.fixed_vars

    def update_fixed_vars(self, fixed_vars):
        self.fixed_vars = fixed_vars


    def __lt__(self, other, strategy="largest_lower_bound"):
        """For priority queue sorting (lower bounds ascending)"""
        if strategy == "largest_lower_bound":
            return self.lower_bound > other.lower_bound
        if strategy == "depth_first":
            return self.level > other.level
        if strategy == "breadth_first":
            return self.level < other.level
        else:
            raise ValueError("Unknown strategy")

class BeB_JS_ID():
    """
    General purpose class for the branch and bound problem
    """
    def __init__(self, P, n_machines, epsilon=0, timelimit = float('inf'), lower_bound_type="linear_relaxation",
                 branching_rule="largest_fractional_job",  node_selection="largest_lower_bound",
                 rounding_rule = "arbitrary_rounding", tol = 1e-5, verbose = 0):
        """
        Parameters:
        - P: the processing times of the jobs
        - n_machines: the number of machines
        - epsilon: the optimality gap
        - timelimit: the time limit for the solver
        - lower_bound_type: the type of lower bound to use
        - branching_rule: the branching rule to use
        - node_selection: the node selection rule to use
        - rounding_rule: the rounding rule to use
        - tol: the tolerance for the rounding rule
        - verbose: the verbosity level. 0 is no output, 1 is some output, 2 is detailed output
        """
        self.epsilon = epsilon
        self.lower_bound_type = lower_bound_type
        self.branching_rule = branching_rule
        self.node_selection = node_selection
        self.timelimit = timelimit

        # Global upper bound and lower bound
        self.GU = float("inf")
        self.GU_argmin = None

        self.GL = float("-inf")

        # Othe parameters
        self.tol = tol
        self.P = P
        self.n_machines = n_machines
        self.rounding_rule = rounding_rule
        self.verbose = verbose

        if self.epsilon < 0:
            raise ValueError("The epsilon must be non-negative")

        if self.lower_bound_type not in ["linear_relaxation", "binary_search"]:
            raise ValueError("The lowerbound must be either 'linear_relaxation' or 'binary_search'")

        if self.branching_rule not in ["largest_fractional_job", "largest_fraction"]:
            raise ValueError("The branching rule must be either 'default' or 'largest_fractional_job'")

        if self.node_selection not in ["largest_lower_bound", "depth_first", "breadth_first"]:
            raise ValueError("The node selection rule must be either 'largest_lower_bound', 'depth_first' or 'breadth_first'")

    def compute_lower_bound(self, P, n_machines, fixed_assignments = None):
        if self.lower_bound_type == "linear_relaxation":
            return linear_relaxation(P, n_machines, check_feasibility=0, fixed_assignments = fixed_assignments)
        if self.lower_bound_type == "binary_search":
            return binary_search(P, n_machines, fixed_assignments=fixed_assignments)
        else:
            raise ValueError("Unknown lower bound")


    def branch(self, X_frac, P):
        fractional_variables = set()
        for (j, i) in X_frac.keys():
            if abs(round(X_frac[j, i]) - X_frac[j, i]) > self.tol:
                fractional_variables.add(j)
        # Cast it to a list
        fractional_variables = list(fractional_variables)
        if self.branching_rule == "largest_fractional_job":
            # Get the one with the largest completion time
            i = max(fractional_variables, key=lambda j: P[j])
        elif self.branching_rule == "largest_fraction":
            # Get the key maximizing the fractional value
            i, _ = max({(i, j) : X_frac[(i, j)] for (i, j) in X_frac.keys() if i in fractional_variables}, key=X_frac.get)
        else:
            raise ValueError("Unknown branching rule, must be either 'largest_fractional_job' or 'largest_fraction', received ", self.branching_rule)
        return i




    def solve(self):
        """
        Solve the branch and bound problem

        Returns:
        - The makespan
        - The assignment
        - The runtime
        - The depth of the tree
        """
        # Start the timer
        start = time.time()

        # Solve the lower bound
        depth = 0
        T_LB, X_frac, status = self.compute_lower_bound(self.P, self.n_machines, fixed_assignments=[(0, 0)]) # Assign the first job to the first machine (w.l.o.g.)

        # Create Node
        root = Node(depth, T_LB, X_frac, fixed_vars=[(0, 0)], rounding_rule=self.rounding_rule)
        # Update the upper bound
        root.set_upper_bound(self.P, self.n_machines)

        # Is it integer?
        if is_integer_sol(X_frac, tol=self.tol):
            return T_LB, X_frac, time.time() - start, depth

        # Update the global values
        self.GU = root.upper_bound
        self.GU_argmin = root.X_int
        self.GL = root.lower_bound

        if self.verbose >= 2:
            print("Initial lower bound: ", T_LB)
            print("Initial upper bound: ", self.GU)

        # Create a priority queue
        queue = []
        # Push the root node in the queue
        heapq.heappush(queue, root)

        # While the queue is not empty
        while len(queue) > 0:
            # Pop the node with the largest lower bound
            node = heapq.heappop(queue)



        # If the queue is empty, return the best solution found so far
        return self.GU, self.GU_argmin, time.time() - start, depth