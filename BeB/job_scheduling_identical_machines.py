from lower_bounds.lower_bounds_JS_identical import linear_relaxation, binary_search
import heapq
from utils import is_integer_sol
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

        # This will be updated in the init
        self.X_int = None
        self.upper_bound = None # This will be set by the solver

        if rounding_rule == "arbitrary_rounding":
            # Just round each job to the machine with the largest fraction
            keys_not_integer = {}
            for (j, i) in X_frac.keys():
                if not abs(round(X_frac[j, i]) - X_frac[j, i]):
                    if j not in keys_not_integer:
                        keys_not_integer[j] = []
                    keys_not_integer[j].append(i)
            X_int = X_frac.copy()
            for j in keys_not_integer:
                i_bar = max(keys_not_integer[j], key=lambda i: X_frac[j, i])
                for (j, i) in X_frac.keys():
                    if i_bar != i:
                        X_int[j, i] = 0
                    else:
                        X_int[j, i] = 1
            # Remove from X_int all the keys for which the value is 0
            X_int = {k: v for k, v in X_int.items() if v > 0}
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
        if self.branching_rule == "largest_fractional_job":
            # Cast it to a list
            fractional_variables = list(fractional_variables)
            # Get the one with the largest completion time
            i = max(fractional_variables, key=lambda j: P[j])
        if self.branching_rule == "largest_fraction":
            i = max(fractional_variables, key=lambda j: X_frac[j])
        else:
            raise ValueError("Unknown branching rule, must be either 'largest_fractional_job' or 'largest_fraction'")
        return i




    def solve(self):
        # Just give some convenient names
        P = self.P
        n_machines = self.n_machines

        # Start a timer
        start = time.time()
        # Initialize the node queue (priority queue based on lower bound)
        node_queue = []

        T, X, feas = self.compute_lower_bound(P, n_machines, fixed_assignments = [(0, 0)]) # The first job is always assigned to the first machine
        assert feas
        if is_integer_sol(X):
            return T, X, time.time() - start, 0
        depth = 0
        root_node = Node(depth, -T, X)
        root_node.set_upper_bound(P, n_machines)

        # Update the global upper bound and the corresponding solution, as well as the global lower bound
        self.GU, self.GU_argmin = root_node.get_upper_bound()
        self.GL = -root_node.lower_bound

        # Push the root node on the queue
        heapq.heappush(node_queue, root_node)



        while len(node_queue) > 0 and time.time() - start < self.timelimit:
            # Update GL: it's the minimal local lower bound of the active nodes
            self.GL = -node_queue[0].lower_bound
            if self.verbose == 1:
                print("STATUS:")
                print("\tcurrent depth: ", depth)
                print("\tCurrent largest lower bound: ", self.GL)
                print("\tCurrent global upper bound: ", self.GU)

            # Get the node with the largest lower bound
            current_node = heapq.heappop(node_queue)

            # First, is the termination criterion satisfied?
            if self.GU / self.GL <= 1 + self.epsilon:
                return self.GU, self.GU_argmin, time.time() - start, depth

            if -current_node.lower_bound >= self.GU: # Don't forget that is negative
                continue # Suboptimal for sure
            else:
                X_fractional_this_node = current_node.get_fractional_solution()
                # If it's not fractional
                if is_integer_sol(X_fractional_this_node):
                    # Check if it's better than the best solution
                    self.GU, self.GU_argmin = -current_node.lower_bound, X_fractional_this_node # That is asctually the integer solution
                    if self.verbose == 1:
                        print("New best solution found: ", self.GU, "in ", time.time() - start, "seconds")

                    # Else you just skip the node -- prune by bound
                else: # Here, the solutios is not integer and the lowerbound is smaller than the best solution, so you have to branch
                    # Branch on the variable that has the largest job
                    j_to_branch = self.branch(X_fractional_this_node, P)
                    fixed_vars = current_node.get_fixed_vars()
                    for i in range(n_machines):
                        new_fixed_vars = fixed_vars + [(j_to_branch, i)]
                        T_new, X_new, _ = self.compute_lower_bound(P, n_machines, fixed_assignments=new_fixed_vars)
                        # Create a new node with all the parameter in places
                        node = Node(depth, -T_new, X_new, rounding_rule=self.rounding_rule)
                        node.update_fixed_vars(new_fixed_vars)
                        node.set_upper_bound(P, n_machines)
                        # Push it on the queue
                        heapq.heappush(node_queue, node)
            # Update the depth
            depth += 1

        # If the queue is empty, return the best solution found so far
        return self.GU, self.GU_argmin, time.time() - start, depth