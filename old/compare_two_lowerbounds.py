import time
import heapq
import os
import numpy as np
from algorithms import JS_ILP
from utils import round_LP_solution_matching
from old.parse_files import parse_instance
import gurobipy as gp
from algorithms import BeB_standard

def list_scheduling_algorithm_identical_with_fixed_jobs(P, n_machines = None, fixed = []):
    '''
     # jobs are arriving ony by one;
    # you assign each of them to the machine where you can start processing them the earliest
    # (e.g. the one whose completion time so far is the smallest)
    :param P:
    :param n_machines:
    :return:
    '''
    n_jobs = P.shape[0]
    if n_machines == None:
        raise ValueError("You must specify the number of machines in this case")
    X = np.zeros((n_jobs, n_machines))
    T = np.zeros(n_machines)

    jobs_assigned = []
    if len(fixed) > 0:
        for (i, j) in fixed:
            T[j] += P[i, 0]
            X[i, j] = 1
            jobs_assigned.append(i)

    for i in range(n_jobs):
        if i not in jobs_assigned:
            # Find the machine with the smallest completion time
            j = np.argmin(T)
            X[i, j] = 1
            T[j] += P[i, 0]
    return max(T), X

def new_LB(P, n_machines, fixed = []):
    # Compute overheads
    T = np.zeros(n_machines)
    X = np.zeros((len(P), n_machines))
    for (i, j) in fixed:
        T[j] += P[i][0].astype(float)
        X[i, j] = 1

    # Take the maximum
    T_max = np.max(T)

    # Compute a stupid lowerbound
    LB = sum(P) / n_machines

    # Get the jobs already assigned
    job_assigned = [i for (i, j) in fixed]

    # Get the maximum and the index of the maximum -- can be improved
    var_to_branch_on = None
    P_current = 0
    for i in range(n_jobs):
        if i not in job_assigned and P[i] >= P_current:
            var_to_branch_on = i
            P_current = P[i]


    if T_max > LB:
        T_int, X = list_scheduling_algorithm_identical_with_fixed_jobs(P, n_machines=n_machines, fixed=fixed)
        # Get the index of the max among the unfixed jobs
        return T_max, X, T_int, var_to_branch_on
    else:
        # Define an LP for job scheduling
        m = gp.Model("job_scheduling")
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0

        # Define a starting budget for all the machined
        b = np.zeros(n_machines)
        fixed_jobs = []
        for (i, j) in fixed:
            b[j] += P[i]
            fixed_jobs.append(i)
        jobs_to_be_assigned = [i for i in range(len(P)) if i not in fixed_jobs]
        X_var = m.addVars(len(jobs_to_be_assigned), n_machines, vtype=gp.GRB.CONTINUOUS, name="X")
        # Each job must be assigned to exactly one machine
        m.addConstrs((gp.quicksum(X_var[i, j] for j in range(n_machines)) == 1 for i in range(len(jobs_to_be_assigned))))
        # On each machine, the total completion time must be less than the lower bound
        m.addConstrs((gp.quicksum(X_var[i, j] * P[jobs_to_be_assigned[i]] for i in range(len(jobs_to_be_assigned))) <= LB - b[j] for j in range(n_machines)))

        # Recover the solution
        m.optimize()

        for i in range(len(jobs_to_be_assigned)):
            for j in range(n_machines):
                X[jobs_to_be_assigned[i], j] = X_var[i, j].x

        # Make this fractional solution integer
        X_int = round_LP_solution_matching(X, P, n_machines)

        # Compute the completion time
        T_int = np.zeros(n_machines)
        for i in range(len(P)):
            for j in range(n_machines):
                T_int[j] += P[i] * X_int[i, j]

        return LB, X_int, max(T_int), var_to_branch_on

class Node:
    def __init__(self, level, bound, solution, upper_bound, variable_to_branch_on, fixed_variables = []):
        self.level = level       # Depth of the node in the tree
        self.lower_bound = bound       # Lower bound for the node
        self.solution = solution # Partial solution up to this node
        self.upper_bound = upper_bound
        self.variable_to_branch_on = variable_to_branch_on
        self.fixed_variables = fixed_variables

    # Comparison operators for priority queue
    def __lt__(self, other):
        return self.lower_bound > other.lower_bound

class BranchAndBound:
    def __init__(self, P, n_machines, timelimit = None, epsilon = 0, verbose= True):
        self.P = P
        self.n_machines = n_machines
        self.best_solution = None
        self.best_value = float('inf')
        self.node_count = 0
        self.best_lower_bound = -float('inf')
        self.runtime = 0
        self.timelimit = timelimit
        self.epsilon = epsilon
        self.verbose = verbose

    def solve(self):
        start = time.time()
        queue = []
        # Build the initial node
        LB, X_int, T_int , var_to_branch_on = new_LB(P, n_machines, fixed = [])
        initial_node = Node(0, LB, X_int, T_int, var_to_branch_on)
        heapq.heappush(queue, initial_node)

        while queue and self.runtime < self.timelimit:

            self.best_lower_bound = queue[0].lower_bound

            current_node = queue.pop(0)

            fixed_vars = current_node.fixed_variables

            # Update the fixed variables
            for j in range(n_machines):
                new_fixed_vars = fixed_vars + [(current_node.variable_to_branch_on, j)]

                # Solve an optimization problem with the new fixed variables
                LB, X_int, T_int, var_to_branch_on = new_LB(P, n_machines, fixed = new_fixed_vars)

                # If the new lower bound is > than the best solution, discard the node (prune by bound)
                if LB > self.best_value:
                    continue
                else:
                    # Add the node to the queue
                    new_node = Node(current_node.level + 1, LB, X_int, T_int, var_to_branch_on, fixed_variables = new_fixed_vars)
                    heapq.heappush(queue, new_node)

                    if T_int < self.best_value:
                        self.best_value = T_int
                        self.best_solution = X_int

            # Control on the quality of the solution
            if self.best_value / self.best_lower_bound <= (1 + self.epsilon):
                print("done by ratio: ", self.best_value / self.best_lower_bound, flush=True)
                return self.best_solution, self.best_value, self.best_lower_bound, self.node_count, self.runtime

            self.node_count += 1
            self.runtime = time.time() - start
            if self.verbose:
                print("Node count: ", self.node_count, "Runtime: ", self.runtime, "Best value: ", self.best_value, "Best lower bound: ", self.best_lower_bound, flush=True)
                print("Queue length: ", len(queue), flush=True)
        if self.runtime >= self.timelimit:
            print("Done by timelimit", flush=True)
        return self.best_solution, self.best_value, self.best_lower_bound, self.node_count, self.runtime

test = True
if test:
    epsilon = 0.01
    dataset = "instancias1a100"

directory_name = "./data/{}".format(dataset)

instances = os.listdir(directory_name)

# Sort by name
instances.sort()

#instances = ['1013.txt']
# Sample a random integer
random_instance = np.random.randint(0, len(instances))
instances = [instances[random_instance]]
print("Instance: ", instances[0], flush=True)

timelimit = 20 # seconds

print("Timelimit: ", timelimit, "seconds", flush=True)

for instance in instances:
    P = parse_instance(instance, directory_name)
    n_jobs, n_machines = P.shape
    # Make the P a column vector -- we are in the identical case
    P = P[:, 0].reshape(-1, 1)

    start = time.time()
    best_objective, best_lb, best_solution, nodes_explored, depth, runtime, optimal = BeB_standard(P, epsilon,
                                                                                                   n_machines,
                                                                                                   timelimit=timelimit)
    print("Our with our B&B, old strategy: ", best_objective, flush=True)
    print("\tTime: ", runtime)
    print("\tNodes: ", nodes_explored)

    beb = BranchAndBound(P, n_machines, epsilon= epsilon, timelimit = timelimit, verbose=False)
    X, best, LB, node_count, runtime = beb.solve()
    print("Best with our B&B, new strategy: ", best, flush=True)
    print("\tTime: ", runtime)
    print("\tNodes: ", bb_nodes)

    # Optimal solution
    opt, X, bb_nodes, runtime, _, _ = JS_ILP(P, n_machines=n_machines, timelimit=timelimit)
    print("Optimal: ", opt, flush=True)
    print("\tTime: ", runtime)
    print("\tNodes: ", bb_nodes)

    if runtime < timelimit:
        print("The optimal solution was found by Gurobi", flush=True)
        print("Ratio: ", best / opt, "<", 1 + epsilon, "?", best / opt < (1 + epsilon), flush=True)