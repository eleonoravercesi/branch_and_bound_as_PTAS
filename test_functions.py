import numpy as np
from BeB.job_scheduling_identical_machines import BeB_JS_ID

n_machines = 2
P = np.array([45, 48, 65, 68, 68, 10, 84, 22, 37, 88])
test =  (0.001, 'linear_relaxation', 'largest_fraction', 'largest_lower_bound', 'arbitrary_rounding')
epsilon, lower_bound_type, branching_rule, node_selection, rounding_rule = test

beb = BeB_JS_ID(P, n_machines, timelimit=600, epsilon=epsilon, lower_bound_type=lower_bound_type,
                                    branching_rule=branching_rule, node_selection=node_selection,
                                    rounding_rule=rounding_rule, verbose=0)
out = beb.solve()

