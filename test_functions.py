import numpy as np
from BeB.job_scheduling_identical_machines import BeB_JS_ID
seed = 0
np.random.seed(seed)
n_machines = 5
jobs = 10
epsilon = 0.001
P = np.random.randint(1, 100, (jobs,))
test =  (0.01, 'binary_search', 'largest_fractional_job', 'largest_lower_bound', 'arbitrary_rounding')
_, lower_bound_type, branching_rule, node_selection, rounding_rule = test

beb = BeB_JS_ID(P, n_machines, timelimit=600, epsilon=epsilon, lower_bound_type=lower_bound_type,
                                    branching_rule=branching_rule, node_selection=node_selection,
                                    rounding_rule=rounding_rule, verbose=0)
makespan, X, time, depth = beb.solve()

makespan_recomputed = {k : 0 for k in range(n_machines)}

for (i, j) in X.keys():
    makespan_recomputed[j] += P[i]

print("Makespan: ", makespan)
print("Makespan recomputed: ", max(makespan_recomputed.values()))