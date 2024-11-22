import numpy as np

def parse_instance(instance_name, dir):
    with open(dir + "/" + instance_name, "r") as f:
        lines = f.readlines()

    # Skip the first two lines
    info = lines[0]
    n_items, n_machines = list(map(int, info.strip().split()[:-1])) # Items, machines
    lines = lines[2:]

    # Define a matrix
    # i, j -> i-th item, j-th machine
    T = np.zeros((n_items,n_machines))
    n_idx = 0
    for line in lines:
        line_vec = line.strip().split()
        # Only the odd entries are relevant
        times = [int(x) for x in line_vec[1::2]]
        T[n_idx, :] = times
        n_idx += 1
    return T

