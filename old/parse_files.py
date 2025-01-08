'''
Parser for problem we want to analyze
For the job scheduling problem, we have this benchmark: TODO disappeared from the web?
For the knapsack, we use instead TODO ?
'''

import numpy as np

def parse_job_scheduling(instance_name, dir, identical=False, uniform=[]):
    '''
    Parse the job scheduling instance
    Parameters
    ----------
    instance_name : str
        The name of the instance
    dir : str
        The directory where the instance is stored
    identical : bool
        If True, the processing times are identical, hence just pick the first column
    uniform : list
        If len(uniform)> 0, the processing times are uniform between the machines, hence just pick the first column and use the
        uniform values as multiplicative factor
    '''
    if identical and len(uniform) > 0:
        raise ValueError("Both identical and uniform cannot be True")

    with open(dir + "/" + instance_name, "r") as f:
        lines = f.readlines()

    # Skip the first two lines
    info = lines[0]
    n_items, n_machines = list(map(int, info.strip().split()[:-1])) # Items, machines
    lines = lines[2:]

    if 0 < len(uniform) < n_machines:
        raise ValueError("The number of uniform values must be greater or equal to the number of machines")

    # Define a matrix
    # j, i -> j-th item, i-th machine
    T = np.zeros((n_items,n_machines))
    n_idx = 0
    for line in lines:
        line_vec = line.strip().split()
        # Only the odd entries are relevant
        times = [int(x) for x in line_vec[1::2]]
        T[n_idx, :] = times
        n_idx += 1
    if identical:
        return T[:,0], n_machines
    if identical == False and len(uniform) == 0:
        return T, n_machines
    if identical == False and len(uniform) > 0:
        T_start = list(T[:,0])
        # Concatenate the column T_start n_machines times muultiplying by the uniform values
        T_out = []
        for i in uniform:
            T_out += [x * i for x in T_start]
        T_out = np.array(T_out).reshape(n_machines, n_items)
        return T_out.T, n_machines
